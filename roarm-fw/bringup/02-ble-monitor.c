#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

#include "ble_nus.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_idf_version.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define FW "02-ble-monitor"
#define USB_UART UART_NUM_0
#define BUS_UART UART_NUM_1
#define S_TXD GPIO_NUM_19
#define S_RXD GPIO_NUM_18
#define BUS_BAUD 1000000
#define ID_FIRST 11
#define ID_LAST 17
#define INST_READ 2
#define INST_WRITE 3
#define REG_TORQUE_EN 40
#define REG_PRESENT 56
#define PRESENT_LEN 15
#define MAT_N 7
#define MAT_BIN (4 + 1 + MAT_N * 11)

static char ble_line[256];
static int ble_line_n;
static TickType_t last_beat;
static int mat_ms;
static TickType_t mat_last;
static uint8_t mat_hdr;
static uint8_t freed;

static unsigned long now_ms(void)
{
    return (unsigned long)(xTaskGetTickCount() * portTICK_PERIOD_MS);
}

static const char *reset_str(esp_reset_reason_t r)
{
    switch (r) {
    case ESP_RST_POWERON: return "poweron";
    case ESP_RST_SW: return "software";
    case ESP_RST_PANIC: return "panic";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_USB: return "usb";
    default: return "other";
    }
}

static void emit_bytes(const char *s, int n)
{
    int i;

    if (n <= 0) {
        return;
    }
    uart_write_bytes(USB_UART, s, n);
    for (i = 0; i < n; i++) {
        if (s[i] == '\n') {
            if (ble_line_n < (int)sizeof(ble_line) - 1) {
                ble_line[ble_line_n++] = '\n';
            }
            ble_nus_send(ble_line, ble_line_n);
            ble_line_n = 0;
        } else if (ble_line_n < (int)sizeof(ble_line) - 1) {
            ble_line[ble_line_n++] = s[i];
        }
    }
}

static int emitf(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        return n;
    }
    if (n >= (int)sizeof(buf)) {
        n = (int)sizeof(buf) - 1;
    }
    emit_bytes(buf, n);
    return n;
}

#define printf(...) emitf(__VA_ARGS__)

static void bus_init(void)
{
    const uart_config_t cfg = {
        .baud_rate = BUS_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(BUS_UART, 1024, 1024, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(BUS_UART, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(BUS_UART, S_TXD, S_RXD,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

static void mk_pkt(uint8_t *out, int *len, uint8_t id, uint8_t inst,
                   const uint8_t *params, int nparams)
{
    uint8_t sum = 0;
    int last;
    int i;

    out[0] = 0xFF;
    out[1] = 0xFF;
    out[2] = id;
    out[3] = (uint8_t)(nparams + 2);
    out[4] = inst;
    if (nparams) {
        memcpy(out + 5, params, (size_t)nparams);
    }
    last = 4 + nparams;
    for (i = 2; i <= last; i++) {
        sum += out[i];
    }
    out[last + 1] = (uint8_t)(~sum);
    *len = last + 2;
}

static int bus_tx_rx(const uint8_t *tx, int txn, uint8_t *rx, int rxmax, int timeout_ms)
{
    int n;

    uart_flush(BUS_UART);
    uart_write_bytes(BUS_UART, (const char *)tx, txn);
    uart_wait_tx_done(BUS_UART, pdMS_TO_TICKS(5));
    n = uart_read_bytes(BUS_UART, rx, rxmax, pdMS_TO_TICKS(timeout_ms));
    if (n < 0) {
        return 0;
    }
    if (n >= txn && memcmp(rx, tx, (size_t)txn) == 0) {
        n -= txn;
        memmove(rx, rx + txn, (size_t)n);
    }
    return n;
}

static int sts_s15(uint16_t v)
{
    if (v & 0x8000) {
        return -(int)(v & 0x7FFF);
    }
    return (int)v;
}

static int bus_read(int id, uint8_t addr, uint8_t nwant, uint8_t *data)
{
    uint8_t params[2] = { addr, nwant };
    uint8_t tx[10], rx[40];
    int txn = 0;
    int n;

    mk_pkt(tx, &txn, (uint8_t)id, INST_READ, params, 2);
    n = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    if (n < 6 + nwant || rx[0] != 0xFF || rx[1] != 0xFF || rx[2] != (uint8_t)id) {
        return 0;
    }
    memcpy(data, rx + 5, nwant);
    return 1;
}

static int bus_write(int id, uint8_t addr, const uint8_t *data, int n)
{
    uint8_t params[16];
    uint8_t tx[24], rx[24];
    int txn = 0;
    int rn;

    if (n < 1 || n > 14) {
        return 0;
    }
    params[0] = addr;
    memcpy(params + 1, data, (size_t)n);
    mk_pkt(tx, &txn, (uint8_t)id, INST_WRITE, params, n + 1);
    rn = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    return rn >= 6 && rx[0] == 0xFF && rx[1] == 0xFF;
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void tq_off_all(void)
{
    uint8_t z = 0;
    int id;

    for (id = ID_FIRST; id <= ID_LAST; id++) {
        bus_write(id, REG_TORQUE_EN, &z, 1);
    }
    printf("mat tq=0 all\n");
}

static void hex_g1(const uint8_t *p, int n)
{
    static const char H[] = "0123456789abcdef";
    char buf[8 + MAT_BIN * 2];
    int i, k = 0;

    buf[k++] = 'g';
    buf[k++] = '1';
    buf[k++] = ' ';
    for (i = 0; i < n; i++) {
        buf[k++] = H[p[i] >> 4];
        buf[k++] = H[p[i] & 0xf];
    }
    buf[k++] = '\n';
    emit_bytes(buf, k);
}

static void mat_tick(void)
{
    uint8_t bin[MAT_BIN];
    uint8_t d[PRESENT_LEN];
    uint8_t tq;
    uint8_t miss = 0;
    TickType_t t;
    int id, i, k, ok_n = 0;

    if (mat_ms <= 0) {
        return;
    }
    t = xTaskGetTickCount();
    if (mat_last != 0 && (t - mat_last) < pdMS_TO_TICKS(mat_ms)) {
        return;
    }
    mat_last = t;
    put_u32(bin, (uint32_t)now_ms());
    k = 5;
    for (i = 0, id = ID_FIRST; id <= ID_LAST; id++, i++) {
        tq = 0;
        if (!bus_read(id, REG_PRESENT, PRESENT_LEN, d)) {
            miss |= (uint8_t)(1 << i);
            memset(bin + k, 0, 11);
            k += 11;
            continue;
        }
        ok_n++;
        bus_read(id, REG_TORQUE_EN, 1, &tq);
        put_u16(bin + k, (uint16_t)sts_s15((uint16_t)(d[0] | (d[1] << 8))));
        put_u16(bin + k + 2, (uint16_t)sts_s15((uint16_t)(d[2] | (d[3] << 8))));
        put_u16(bin + k + 4, (uint16_t)sts_s15((uint16_t)(d[4] | (d[5] << 8))));
        put_u16(bin + k + 6, (uint16_t)sts_s15((uint16_t)(d[13] | (d[14] << 8))));
        bin[k + 8] = d[6];
        bin[k + 9] = d[7];
        bin[k + 10] = (uint8_t)((tq ? 1 : 0) | (d[10] ? 2 : 0));
        k += 11;
    }
    bin[4] = miss;
    if (ok_n && !freed) {
        tq_off_all();
        freed = 1;
    }
    if (!ok_n) {
        freed = 0;
    }
    hex_g1(bin, MAT_BIN);
}

static void cmd_mat(const char *what)
{
    while (*what && isspace((unsigned char)*what)) {
        what++;
    }
    if (*what == '\0' || strcmp(what, "on") == 0) {
        mat_ms = 50;
    } else if (strcmp(what, "off") == 0) {
        mat_ms = 0;
    } else {
        int ms = atoi(what);
        if (ms < 0) {
            ms = 0;
        }
        if (ms > 0 && ms < 40) {
            ms = 40;
        }
        if (ms > 2000) {
            ms = 2000;
        }
        mat_ms = ms;
    }
    mat_hdr = mat_ms ? 1 : 0;
    mat_last = 0;
    printf("mat ms=%d\n", mat_ms);
}

static void cmd_info(void)
{
    esp_chip_info_t chip;
    uint32_t flash_size = 0;
    uint8_t mac[6];

    esp_chip_info(&chip);
    esp_flash_get_size(NULL, &flash_size);
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("info chip=ESP32 cores=%d rev=%d flash=%luKB reset=%s\n",
           chip.cores, chip.revision,
           (unsigned long)(flash_size / 1024), reset_str(esp_reset_reason()));
    printf("info mac=%02x:%02x:%02x:%02x:%02x:%02x heap=%u uptime_ms=%lu\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
           (unsigned)esp_get_free_heap_size(), now_ms());
    printf("info idf=%s usb=115200 bus=gpio18/19@1Mbps ids=11-17 ble=%s\n",
           esp_get_idf_version(), ble_nus_name());
    printf("info fw=%s raise=0 stream=0 mat=1 guard=0\n", FW);
    printf("info board=gibbon-%02x%02x\n", mac[4], mac[5]);
    printf("info note=monitor only. no move. unplug USB after flash; pack feeds BLE.\n");
}

static void handle_line(char *line)
{
    while (*line && isspace((unsigned char)*line)) {
        line++;
    }
    if (*line == '\0') {
        return;
    }
    if (strcmp(line, "help") == 0) {
        printf("commands: help info mat [N|off] estop\n");
        return;
    }
    if (strcmp(line, "info") == 0) {
        cmd_info();
        return;
    }
    if (strcmp(line, "mat") == 0) {
        cmd_mat("");
        return;
    }
    if (strncmp(line, "mat ", 4) == 0) {
        cmd_mat(line + 4);
        return;
    }
    if (strcmp(line, "estop") == 0) {
        tq_off_all();
        return;
    }
    printf("err unknown '%s'\n", line);
}

void app_main(void)
{
    const uart_config_t usb = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uint8_t mac[6];
    char line[64];
    int line_len = 0;

    ESP_ERROR_CHECK(uart_param_config(USB_UART, &usb));
    ESP_ERROR_CHECK(uart_driver_install(USB_UART, 1024, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(USB_UART);
    bus_init();
    last_beat = xTaskGetTickCount();
    ble_nus_init();
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("\nbringup %s\n", FW);
    printf("mac=%02x:%02x:%02x:%02x:%02x:%02x\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("note=no move. BLE matrix 50ms after subscribe. unplug USB.\n");

    for (;;) {
        char bline[64];

        ble_nus_tick();
        if (ble_nus_subscribed() && mat_ms == 0) {
            mat_ms = 50;
            mat_last = 0;
        }
        mat_tick();
        while (ble_nus_pop_line(bline, (int)sizeof(bline))) {
            handle_line(bline);
        }
        if (mat_ms == 0) {
            TickType_t t = xTaskGetTickCount();
            if ((t - last_beat) >= pdMS_TO_TICKS(2000)) {
                last_beat = t;
                printf("t=%lu heap=%u ble=%d mat=%d\n",
                       now_ms(), (unsigned)esp_get_free_heap_size(),
                       ble_nus_connected(), mat_ms);
            }
        }
        {
            uint8_t ch;
            int n = uart_read_bytes(USB_UART, &ch, 1, pdMS_TO_TICKS(10));
            if (n != 1) {
                continue;
            }
            if (ch == '\r' || ch == '\n') {
                if (line_len > 0) {
                    line[line_len] = '\0';
                    handle_line(line);
                    line_len = 0;
                }
            } else if (line_len < (int)sizeof(line) - 1) {
                line[line_len++] = (char)ch;
            }
        }
    }
}
