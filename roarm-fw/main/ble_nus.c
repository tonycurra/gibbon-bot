#include "ble_nus.h"

#include <stdio.h>
#include <string.h>

#include "esp_mac.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "host/ble_att.h"
#include "host/ble_gap.h"
#include "host/ble_hs.h"
#include "host/ble_sm.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define NUS_LINE_MAX 64
#define NUS_RXQ_LEN 8

/* 6e400001-b5a3-f393-e0a9-e50e24dcca9e (little-endian bytes) */
static const ble_uuid128_t nus_svc_uuid =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x01, 0x00, 0x40, 0x6e);
static const ble_uuid128_t nus_rx_uuid =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x02, 0x00, 0x40, 0x6e);
static const ble_uuid128_t nus_tx_uuid =
    BLE_UUID128_INIT(0x9e, 0xca, 0xdc, 0x24, 0x0e, 0xe5, 0xa9, 0xe0,
                     0x93, 0xf3, 0xa3, 0xb5, 0x03, 0x00, 0x40, 0x6e);

void ble_store_config_init(void);

static uint16_t tx_handle;
static uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint8_t own_addr_type;
static volatile int subscribed;
static TickType_t conn_since;
static uint16_t att_mtu = 23;
static char gap_name[16];
static QueueHandle_t rxq;
static char rx_acc[NUS_LINE_MAX];
static int rx_acc_n;

static int gap_event(struct ble_gap_event *event, void *arg);
static int gatt_access(uint16_t conn, uint16_t attr,
                       struct ble_gatt_access_ctxt *ctxt, void *arg);

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &nus_svc_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &nus_rx_uuid.u,
                .access_cb = gatt_access,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            {
                .uuid = &nus_tx_uuid.u,
                .access_cb = gatt_access,
                .val_handle = &tx_handle,
                .flags = BLE_GATT_CHR_F_NOTIFY | BLE_GATT_CHR_F_READ,
            },
            { 0 },
        },
    },
    { 0 },
};

static void rx_bytes(const uint8_t *p, int n)
{
    for (int i = 0; i < n; i++) {
        char c = (char)p[i];
        if (c == '\r') {
            continue;
        }
        if (c == '\n') {
            if (rx_acc_n > 0 && rxq) {
                char line[NUS_LINE_MAX];
                memcpy(line, rx_acc, (size_t)rx_acc_n);
                line[rx_acc_n] = '\0';
                xQueueSend(rxq, line, 0);
            }
            rx_acc_n = 0;
            continue;
        }
        if (rx_acc_n < NUS_LINE_MAX - 1) {
            rx_acc[rx_acc_n++] = c;
        }
    }
}

static int gatt_access(uint16_t conn, uint16_t attr,
                       struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn;
    (void)attr;
    (void)arg;

    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        uint16_t len = OS_MBUF_PKTLEN(ctxt->om);
        uint8_t buf[128];
        if (len > sizeof(buf)) {
            len = sizeof(buf);
        }
        if (ble_hs_mbuf_to_flat(ctxt->om, buf, len, NULL) == 0) {
            rx_bytes(buf, (int)len);
        }
        return 0;
    }
    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        return os_mbuf_append(ctxt->om, "", 0) == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
    }
    return 0;
}

static void advertise(void)
{
    struct ble_hs_adv_fields fields;
    struct ble_hs_adv_fields rsp;
    struct ble_gap_adv_params adv;
    static ble_uuid128_t adv_uuid;
    const char *name = ble_svc_gap_device_name();
    static const char short_name[] = "gibbon";
    int rc;

    /* ADV: flags + NUS UUID + short name. Fits in 31 B so Chrome can filter
       by service or namePrefix without waiting on a scan response. */
    adv_uuid = nus_svc_uuid;
    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.uuids128 = &adv_uuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;
    fields.name = (uint8_t *)short_name;
    fields.name_len = sizeof(short_name) - 1;
    fields.name_is_complete = 0;
    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        printf("ble adv fields rc=%d\n", rc);
        return;
    }

    memset(&rsp, 0, sizeof(rsp));
    rsp.name = (uint8_t *)name;
    rsp.name_len = strlen(name);
    rsp.name_is_complete = 1;
    rc = ble_gap_adv_rsp_set_fields(&rsp);
    if (rc != 0) {
        printf("ble scan rsp rc=%d\n", rc);
    }

    memset(&adv, 0, sizeof(adv));
    adv.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv.disc_mode = BLE_GAP_DISC_MODE_GEN;
    /* 0.625 ms units. 160–320 ms — findable without a hot radio. */
    adv.itvl_min = 0x100;
    adv.itvl_max = 0x200;
    rc = ble_gap_adv_start(own_addr_type, NULL, BLE_HS_FOREVER,
                           &adv, gap_event, NULL);
    if (rc != 0) {
        printf("ble adv start rc=%d\n", rc);
        return;
    }
    printf("ble advertising %s\n", name);
}

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            conn_handle = event->connect.conn_handle;
            att_mtu = 23;
            subscribed = 0;
            conn_since = xTaskGetTickCount();
            {
                struct ble_gap_upd_params p = {
                    .itvl_min = 24,
                    .itvl_max = 40,
                    .latency = 2,
                    .supervision_timeout = 200,
                    .min_ce_len = 0,
                    .max_ce_len = 0,
                };
                ble_gap_update_params(conn_handle, &p);
            }
            printf("ble connected handle=%d\n", conn_handle);
        } else {
            conn_handle = BLE_HS_CONN_HANDLE_NONE;
            subscribed = 0;
            advertise();
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        printf("ble disconnect reason=%d\n", event->disconnect.reason);
        conn_handle = BLE_HS_CONN_HANDLE_NONE;
        subscribed = 0;
        att_mtu = 23;
        advertise();
        return 0;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == tx_handle) {
            subscribed = event->subscribe.cur_notify ? 1 : 0;
        }
        return 0;

    case BLE_GAP_EVENT_MTU:
        att_mtu = event->mtu.value;
        printf("ble mtu=%d\n", att_mtu);
        return 0;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        advertise();
        return 0;

    default:
        return 0;
    }
}

static void on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc != 0) {
        printf("ble addr rc=%d\n", rc);
        return;
    }
    rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc != 0) {
        printf("ble id rc=%d\n", rc);
        return;
    }
    advertise();
}

static void on_reset(int reason)
{
    printf("ble reset reason=%d\n", reason);
}

static void host_task(void *param)
{
    (void)param;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static int gatt_init(void)
{
    int rc;

    ble_svc_gap_init();
    ble_svc_gatt_init();
    rc = ble_gatts_count_cfg(gatt_svcs);
    if (rc != 0) {
        return rc;
    }
    return ble_gatts_add_svcs(gatt_svcs);
}

void ble_nus_init(void)
{
    uint8_t mac[6];
    esp_err_t err;
    int rc;

    rxq = xQueueCreate(NUS_RXQ_LEN, NUS_LINE_MAX);
    if (!rxq) {
        printf("ble rxq fail\n");
        return;
    }

    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        printf("ble nvs rc=%d\n", (int)err);
        return;
    }

    err = nimble_port_init();
    if (err != ESP_OK) {
        printf("ble nimble init rc=%d\n", (int)err);
        return;
    }

    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    snprintf(gap_name, sizeof(gap_name), "gibbon-%02x%02x", mac[4], mac[5]);

    ble_hs_cfg.reset_cb = on_reset;
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;
    ble_hs_cfg.sm_io_cap = BLE_SM_IO_CAP_NO_IO;
    ble_hs_cfg.sm_bonding = 0;
    ble_hs_cfg.sm_mitm = 0;
    ble_hs_cfg.sm_sc = 0;
    ble_store_config_init();

    rc = gatt_init();
    if (rc != 0) {
        printf("ble gatt rc=%d\n", rc);
        return;
    }
    ble_att_set_preferred_mtu(247);
    ble_svc_gap_device_name_set(gap_name);
    nimble_port_freertos_init(host_task);
    printf("ble name=%s nus=nordic-uart\n", gap_name);
}

int ble_nus_connected(void)
{
    return conn_handle != BLE_HS_CONN_HANDLE_NONE;
}

int ble_nus_subscribed(void)
{
    return subscribed;
}

void ble_nus_tick(void)
{
    if (conn_handle == BLE_HS_CONN_HANDLE_NONE || subscribed) {
        return;
    }
    if ((xTaskGetTickCount() - conn_since) > pdMS_TO_TICKS(5000)) {
        printf("ble ghost — drop\n");
        ble_gap_terminate(conn_handle, 0x13);
        conn_since = xTaskGetTickCount();
    }
}

const char *ble_nus_name(void)
{
    return gap_name[0] ? gap_name : "gibbon";
}

void ble_nus_send(const void *data, int n)
{
    const uint8_t *p = data;
    int chunk;

    if (!subscribed || conn_handle == BLE_HS_CONN_HANDLE_NONE || n <= 0) {
        return;
    }
    chunk = (int)att_mtu - 3;
    if (chunk < 20) {
        chunk = 20;
    }
    while (n > 0) {
        int k = n < chunk ? n : chunk;
        struct os_mbuf *om = ble_hs_mbuf_from_flat(p, k);
        int rc;

        if (!om) {
            return;
        }
        rc = ble_gatts_notify_custom(conn_handle, tx_handle, om);
        if (rc != 0) {
            return;
        }
        p += k;
        n -= k;
    }
}

int ble_nus_pop_line(char *out, int max)
{
    char line[NUS_LINE_MAX];

    if (!rxq || max < 2) {
        return 0;
    }
    if (xQueueReceive(rxq, line, 0) != pdTRUE) {
        return 0;
    }
    strncpy(out, line, (size_t)max - 1);
    out[max - 1] = '\0';
    return 1;
}
