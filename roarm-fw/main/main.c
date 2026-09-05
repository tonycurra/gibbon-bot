#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

#define USB_UART UART_NUM_0
#define BUS_UART UART_NUM_1
#define S_TXD GPIO_NUM_19
#define S_RXD GPIO_NUM_18
#define BUS_BAUD 1000000
#define ID_FIRST 11
#define ID_LAST 17

#define INST_PING 1
#define INST_READ 2
#define INST_WRITE 3
#define REG_TORQUE_EN 40
#define REG_ACC 41
#define REG_PRESENT 56
#define PRESENT_LEN 15 /* pos..current at 69 */
#define REG_VOLTAGE 62
#define HAND_ID 17
#define HAND_OPEN 689
#define HAND_CLOSE 2008
#define HAND_TO_MAX 2075 /* inside wall 2088 — contact probe, not the wall */
#define HAND_SPEED 300
#define HAND_ACC 20
#define V_CUTOFF 105 /* 10.5 V, 0.1 V LSB */

static int listen_dump = 1;
static uint32_t bus_rx_total;
static uint8_t acc[64];
static int acc_n;
static TickType_t acc_last;
static TickType_t last_beat;

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
    case ESP_RST_INT_WDT: return "int_wdt";
    case ESP_RST_TASK_WDT: return "task_wdt";
    case ESP_RST_WDT: return "wdt";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_USB: return "usb";
    default: return "other";
    }
}

static void print_help(void)
{
    printf("commands:\n");
    printf("  (nothing) heartbeats + raw UART1 RX. no move.\n");
    printf("  state    chip / tasks / heap / bus_rx\n");
    printf("  listen   RX hex dump on/off\n");
    printf("  gpio     levels on 18/19 (still UART, not remuxed)\n");
    printf("  ping     IDs 11-17 (read-only ping, still no move)\n");
    printf("  pingall  IDs 1-253, hits only\n");
    printf("  raw ID   ping + READ present, dump hex\n");
    printf("  get      pos/speed/load/V/temp/current/moving/torque-en (READ only)\n");
    printf("  batt     pack V from servo VIN + 3S rest-%% guess (not a fuel gauge)\n");
    printf("  bench    chip + ping 11-17\n");
    printf("  hand open   ID 17 to %d (soft open). slow. no other ID\n", HAND_OPEN);
    printf("  hand close  ID 17 to %d (soft close). slow. no other ID\n", HAND_CLOSE);
    printf("  hand to N   ID 17 to N, clamped %d-%d (wall 2088)\n", HAND_OPEN, HAND_TO_MAX);
    printf("  hand free   ID 17 torque off\n");
    printf("  help\n");
}

static void print_hex(const uint8_t *p, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%02x%s", p[i], (i + 1 < n) ? " " : "");
    }
}

static void dump_bus(const uint8_t *p, int n)
{
    printf("bus t=%lu n=%d hex=", now_ms(), n);
    print_hex(p, n);
    printf("\n");
}

static void acc_flush(void)
{
    if (acc_n <= 0) {
        return;
    }
    if (listen_dump) {
        dump_bus(acc, acc_n);
    }
    acc_n = 0;
}

static void bus_eat(int dump)
{
    uint8_t buf[64];
    for (;;) {
        int n = uart_read_bytes(BUS_UART, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        bus_rx_total += (uint32_t)n;
        if (!dump) {
            continue;
        }
        acc_last = xTaskGetTickCount();
        for (int i = 0; i < n; i++) {
            if (acc_n >= (int)sizeof(acc)) {
                acc_flush();
            }
            acc[acc_n++] = buf[i];
        }
    }
    if (acc_n > 0 && (xTaskGetTickCount() - acc_last) >= 1) {
        acc_flush();
    }
}

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

static int bus_tx_rx(const uint8_t *tx, int txn, uint8_t *rx, int rxmax, int timeout_ms)
{
    int saved = listen_dump;
    listen_dump = 0;
    acc_n = 0;
    uart_flush(BUS_UART);
    uart_write_bytes(BUS_UART, (const char *)tx, txn);
    uart_wait_tx_done(BUS_UART, pdMS_TO_TICKS(5));
    int n = uart_read_bytes(BUS_UART, rx, rxmax, pdMS_TO_TICKS(timeout_ms));
    listen_dump = saved;
    if (n < 0) {
        return 0;
    }
    if (n >= txn && memcmp(rx, tx, (size_t)txn) == 0) {
        n -= txn;
        memmove(rx, rx + txn, (size_t)n);
    }
    return n;
}

static void mk_pkt(uint8_t *out, int *len, uint8_t id, uint8_t inst,
                   const uint8_t *params, int nparams)
{
    out[0] = 0xFF;
    out[1] = 0xFF;
    out[2] = id;
    out[3] = (uint8_t)(nparams + 2);
    out[4] = inst;
    if (nparams) {
        memcpy(out + 5, params, (size_t)nparams);
    }
    uint8_t sum = 0;
    int last = 4 + nparams;
    for (int i = 2; i <= last; i++) {
        sum += out[i];
    }
    out[last + 1] = (uint8_t)(~sum);
    *len = last + 2;
}

static int bus_ping(int id)
{
    uint8_t tx[8], rx[24];
    int txn = 0;
    mk_pkt(tx, &txn, (uint8_t)id, INST_PING, NULL, 0);
    int n = bus_tx_rx(tx, txn, rx, sizeof(rx), 10);
    if (n < 6) {
        return 0;
    }
    return (rx[0] == 0xFF && rx[1] == 0xFF);
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
    mk_pkt(tx, &txn, (uint8_t)id, INST_READ, params, 2);
    int n = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    if (n < 6 + nwant) {
        return 0;
    }
    int i = 0;
    if (!(rx[0] == 0xFF && rx[1] == 0xFF)) {
        return 0;
    }
    /* status: FF FF ID LEN ERR DATA… */
    if (rx[2] != (uint8_t)id) {
        return 0;
    }
    i = 5;
    if (n - i < nwant) {
        return 0;
    }
    memcpy(data, rx + i, nwant);
    return 1;
}

static int bus_write(int id, uint8_t addr, const uint8_t *data, int n)
{
    uint8_t params[16];
    if (n < 1 || n > 14) {
        return 0;
    }
    params[0] = addr;
    memcpy(params + 1, data, (size_t)n);
    uint8_t tx[24], rx[24];
    int txn = 0;
    mk_pkt(tx, &txn, (uint8_t)id, INST_WRITE, params, n + 1);
    int rn = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    return rn >= 6 && rx[0] == 0xFF && rx[1] == 0xFF;
}

static void cmd_hand(const char *what)
{
    if (strcmp(what, "free") == 0) {
        uint8_t tq = 0;
        if (!bus_write(HAND_ID, REG_TORQUE_EN, &tq, 1)) {
            printf("hand free miss\n");
            return;
        }
        printf("hand free tq=0 id=%d\n", HAND_ID);
        return;
    }

    int goal;
    const char *tag = what;
    int hi = HAND_CLOSE;
    if (strcmp(what, "open") == 0) {
        goal = HAND_OPEN;
    } else if (strcmp(what, "close") == 0) {
        goal = HAND_CLOSE;
    } else if (strncmp(what, "to ", 3) == 0) {
        goal = atoi(what + 3);
        tag = "to";
        hi = HAND_TO_MAX;
    } else {
        printf("err hand open|close|to N|free\n");
        return;
    }
    if (goal < HAND_OPEN) {
        goal = HAND_OPEN;
    }
    if (goal > hi) {
        goal = hi;
    }

    uint8_t v = 0;
    if (!bus_read(HAND_ID, REG_VOLTAGE, 1, &v) || v < V_CUTOFF) {
        printf("hand abort pack v_raw=%d need>=%d\n", v, V_CUTOFF);
        return;
    }

    uint8_t tq = 1;
    if (!bus_write(HAND_ID, REG_TORQUE_EN, &tq, 1)) {
        printf("hand torque miss\n");
        return;
    }

    uint8_t buf[7];
    buf[0] = HAND_ACC;
    buf[1] = (uint8_t)(goal & 0xFF);
    buf[2] = (uint8_t)((goal >> 8) & 0xFF);
    buf[3] = 0;
    buf[4] = 0;
    buf[5] = (uint8_t)(HAND_SPEED & 0xFF);
    buf[6] = (uint8_t)((HAND_SPEED >> 8) & 0xFF);
    if (!bus_write(HAND_ID, REG_ACC, buf, 7)) {
        printf("hand write miss goal=%d\n", goal);
        return;
    }
    printf("hand %s id=%d goal=%d speed=%d acc=%d tq=1 walls=609/2088\n",
           tag, HAND_ID, goal, HAND_SPEED, HAND_ACC);
}

static void cmd_info(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);

    printf("info chip=ESP32 cores=%d rev=%d flash=%luKB reset=%s\n",
           chip.cores, chip.revision,
           (unsigned long)(flash_size / 1024),
           reset_str(esp_reset_reason()));
    printf("info mac=%02x:%02x:%02x:%02x:%02x:%02x heap=%u uptime_ms=%lu\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
           (unsigned)esp_get_free_heap_size(), now_ms());
    printf("info idf=%s usb=115200 bus=gpio18/19@1Mbps ids=11-17\n",
           esp_get_idf_version());
    printf("info note=listen-only until ping/get. no move. servos need XT60\n");
}

static void cmd_state(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("state t=%lu heap=%u reset=%s tasks=%u bus_rx=%lu dump=%d\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           reset_str(esp_reset_reason()),
           (unsigned)uxTaskGetNumberOfTasks(),
           (unsigned long)bus_rx_total,
           listen_dump);
    printf("state mac=%02x:%02x:%02x:%02x:%02x:%02x\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("state bus=uart1 gpio18/19 1Mbps rx-only until ping\n");
#if CONFIG_FREERTOS_USE_TRACE_FACILITY && CONFIG_FREERTOS_USE_STATS_FORMATTING_FUNCTIONS
    {
        static char tasks[768];
        vTaskList(tasks);
        printf("tasks:\n%s", tasks);
    }
#endif
}

static void cmd_gpio(void)
{
    printf("gpio 18(rx)=%d 19(tx)=%d (uart1, not remuxed)\n",
           gpio_get_level(S_RXD), gpio_get_level(S_TXD));
}

static void cmd_ping(int lo, int hi, int verbose)
{
    printf("ping uart1 gpio18/19 1Mbps (UART switch = ESP32)\n");
    int found = 0;
    for (int id = lo; id <= hi; id++) {
        if (bus_ping(id)) {
            printf("ping FOUND id=%d\n", id);
            found++;
        } else if (verbose) {
            printf("ping miss id=%d\n", id);
        }
    }
    if (!found) {
        printf("ping none %d-%d — expected if battery is off\n", lo, hi);
    } else {
        printf("ping done found=%d\n", found);
    }
}

static void cmd_bench(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);

    int bits[8];
    int found = 0;
    int nids = ID_LAST - ID_FIRST + 1;
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        int ok = bus_ping(id);
        bits[id - ID_FIRST] = ok;
        found += ok;
    }

    printf("bench v1\n");
    printf("chip=ESP32 cores=%d rev=%d flash=%luKB\n",
           chip.cores, chip.revision, (unsigned long)(flash_size / 1024));
    printf("mac=%02x:%02x:%02x:%02x:%02x:%02x\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("bus=gpio18/19@1Mbps ids=%d-%d\n", ID_FIRST, ID_LAST);
    printf("ping");
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        printf(" %d=%s", id, bits[id - ID_FIRST] ? "found" : "miss");
    }
    printf("\n");
    printf("found=%d/%d\n", found, nids);
    if (found == 0) {
        printf("guess=usb-no-battery\n");
    } else if (found == nids) {
        printf("guess=usb-battery-on\n");
    } else {
        printf("guess=partial\n");
    }
}

static void cmd_raw(int id)
{
    if (id < 1 || id > 253) {
        printf("err raw id 1-253\n");
        return;
    }
    uint8_t tx[10], rx[40];
    int txn = 0;
    mk_pkt(tx, &txn, (uint8_t)id, INST_PING, NULL, 0);
    int n = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    printf("raw ping id=%d n=%d hex=", id, n);
    print_hex(rx, n);
    printf("\n");

    uint8_t params[2] = { REG_PRESENT, PRESENT_LEN };
    txn = 0;
    mk_pkt(tx, &txn, (uint8_t)id, INST_READ, params, 2);
    n = bus_tx_rx(tx, txn, rx, sizeof(rx), 12);
    printf("raw read id=%d n=%d hex=", id, n);
    print_hex(rx, n);
    printf("\n");
}

static void cmd_get(void)
{
    printf("get present+torque-en. hand open/close is the only write\n");
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        uint8_t d[PRESENT_LEN];
        uint8_t tq = 0;
        if (!bus_read(id, REG_PRESENT, PRESENT_LEN, d)) {
            printf("get id=%d miss\n", id);
            continue;
        }
        int pos = sts_s15((uint16_t)(d[0] | (d[1] << 8)));
        int speed = sts_s15((uint16_t)(d[2] | (d[3] << 8)));
        int load = sts_s15((uint16_t)(d[4] | (d[5] << 8)));
        int v_raw = d[6];
        int temp = d[7];
        int moving = d[10];
        int current = sts_s15((uint16_t)(d[13] | (d[14] << 8)));
        int tq_ok = bus_read(id, REG_TORQUE_EN, 1, &tq);
        printf("get id=%d pos=%d speed=%d load=%d v_raw=%d v=%.1f temp=%d cur=%d moving=%d tq=%s\n",
               id, pos, speed, load, v_raw, v_raw / 10.0, temp, current, moving,
               tq_ok ? (tq ? "1" : "0") : "?");
    }
}

/* 3S LiPo rest SoC from pack millivolts. Coarse. Not charging current. */
static int lipo3s_pct(int mv)
{
    static const int t[][2] = {
        {12600, 100}, {12480, 95}, {12360, 90}, {12210, 80}, {12060, 70},
        {11910, 60}, {11790, 50}, {11670, 40}, {11550, 30}, {11400, 25},
        {11310, 20}, {11100, 15}, {10830, 10}, {10500, 5}, {9900, 0},
    };
    const int n = (int)(sizeof(t) / sizeof(t[0]));
    if (mv >= t[0][0]) {
        return 100;
    }
    if (mv <= t[n - 1][0]) {
        return 0;
    }
    for (int i = 1; i < n; i++) {
        if (mv >= t[i][0]) {
            int mv0 = t[i][0], p0 = t[i][1];
            int mv1 = t[i - 1][0], p1 = t[i - 1][1];
            return p0 + (p1 - p0) * (mv - mv0) / (mv1 - mv0);
        }
    }
    return 0;
}

static void cmd_batt(void)
{
    uint8_t vs[8];
    int n = 0;
    printf("batt READ servo VIN (same rail as XT60). 0.1V LSB. no INA219\n");
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        uint8_t v = 0;
        if (!bus_read(id, REG_VOLTAGE, 1, &v)) {
            printf("batt id=%d miss\n", id);
            continue;
        }
        vs[n++] = v;
        printf("batt id=%d v=%.1f\n", id, v / 10.0);
    }
    if (n == 0) {
        printf("batt pack=miss est=? note=servos need XT60\n");
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (vs[j] < vs[i]) {
                uint8_t tmp = vs[i];
                vs[i] = vs[j];
                vs[j] = tmp;
            }
        }
    }
    int v_raw = vs[n / 2];
    int mv = v_raw * 100;
    int pct = lipo3s_pct(mv);
    const char *lvl = (mv < 10500) ? "cutoff" : (mv < 10800) ? "low" : "ok";
    printf("batt pack=%.1fV est=%d%% cells=3 n=%d %s\n",
           v_raw / 10.0, pct, n, lvl);
    printf("batt note=rest-voltage guess, not a BMS. HAT does not charge XT60; USB is not a charger.\n");
}

static void print_banner(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("\nlisten v1\n");
    printf("mac=%02x:%02x:%02x:%02x:%02x:%02x\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("reset=%s\n", reset_str(esp_reset_reason()));
    printf("tasks=%u\n", (unsigned)uxTaskGetNumberOfTasks());
    printf("bus=uart1 gpio18/19 1Mbps rx-only until ping\n");
    printf("note=no auto-move. hand open/close only, soft 689-2008.\n");
}

static void heartbeat(void)
{
    TickType_t t = xTaskGetTickCount();
    if ((t - last_beat) < pdMS_TO_TICKS(2000)) {
        return;
    }
    last_beat = t;
    printf("t=%lu heap=%u bus_rx=%lu\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           (unsigned long)bus_rx_total);
}

static void handle_line(char *line)
{
    while (*line && isspace((unsigned char)*line)) {
        line++;
    }
    size_t n = strlen(line);
    while (n > 0 && isspace((unsigned char)line[n - 1])) {
        line[--n] = '\0';
    }
    if (n == 0) {
        return;
    }

    if (strcmp(line, "help") == 0) {
        print_help();
        return;
    }
    if (strcmp(line, "info") == 0) {
        cmd_info();
        return;
    }
    if (strcmp(line, "state") == 0) {
        cmd_state();
        return;
    }
    if (strcmp(line, "listen") == 0) {
        listen_dump = !listen_dump;
        printf("listen dump=%d\n", listen_dump);
        return;
    }
    if (strcmp(line, "gpio") == 0) {
        cmd_gpio();
        return;
    }
    if (strcmp(line, "ping") == 0) {
        cmd_ping(ID_FIRST, ID_LAST, 1);
        return;
    }
    if (strcmp(line, "pingall") == 0) {
        cmd_ping(1, 253, 0);
        return;
    }
    if (strcmp(line, "bench") == 0) {
        cmd_bench();
        return;
    }
    if (strcmp(line, "get") == 0) {
        cmd_get();
        return;
    }
    if (strcmp(line, "batt") == 0) {
        cmd_batt();
        return;
    }
    if (strncmp(line, "raw ", 4) == 0) {
        cmd_raw(atoi(line + 4));
        return;
    }
    if (strncmp(line, "hand ", 5) == 0) {
        cmd_hand(line + 5);
        return;
    }

    printf("err unknown '%s' — type help\n", line);
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
    ESP_ERROR_CHECK(uart_param_config(USB_UART, &usb));
    ESP_ERROR_CHECK(uart_driver_install(USB_UART, 1024, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(USB_UART);

    bus_init();
    last_beat = xTaskGetTickCount();

    print_banner();
    print_help();
    cmd_state();

    char line[64];
    int line_len = 0;
    for (;;) {
        bus_eat(1);
        heartbeat();

        uint8_t ch;
        int n = uart_read_bytes(USB_UART, &ch, 1, pdMS_TO_TICKS(20));
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
