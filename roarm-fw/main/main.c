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
#define REG_FW 0
#define REG_MIN_ANGLE 9
#define REG_MAX_TORQUE 16
#define REG_KP 21
#define REG_MODE 33
#define REG_TORQUE_EN 40
#define REG_ACC 41
#define REG_GOAL 42
#define REG_GOAL_SPEED 46
#define REG_LOCK 55
#define REG_PRESENT 56
#define PRESENT_LEN 15 /* pos..current at 69 */
#define REG_VOLTAGE 62
#define HAND_ID 17
#define HAND_OPEN 689
#define HAND_CLOSE 2008
#define HAND_TO_MAX 2075 /* inside wall 2088 — contact probe, not the wall */
#define HAND_SPEED 300
#define HAND_ACC 20
#define TWIST_ID 16
#define TWIST_MIN 0
#define TWIST_MAX 4095
#define TWIST_CENTER 4050 /* wrap-seam home; 4050 and ~50 are the same neighborhood */
#define TWIST_STEP_MAX 900 /* host walks a full turn in small hops; refuse a long hop */
#define NUDGE_MAX 80 /* unwalled IDs 11–15: no long WritePos */
#define V_CUTOFF 105 /* 10.5 V, 0.1 V LSB */
#define GUARD_DLOAD 100 /* |load| above moving cruise — cable tighten */
#define GUARD_LOAD 1350 /* hard |load|; plus-bias cruise is ~1100–1200 */
#define GUARD_CUR 80 /* |mA|; free move was 0–9, stall is amps */
#define GUARD_MISS 3 /* bus miss while pushing → cut */

typedef struct {
    int pos;
    int speed;
    int load;
    int v_raw;
    int temp;
    int moving;
    int cur;
} present_t;

static int listen_dump = 1;
static uint32_t bus_rx_total;
static uint8_t acc[64];
static int acc_n;
static TickType_t acc_last;
static TickType_t last_beat;
static int last_pos[ID_LAST + 1];
static int wrap_turns[ID_LAST + 1];
static uint8_t pos_seen[ID_LAST + 1];
static int twist_guard_on;
static int twist_cruise_n;
static int twist_cruise_load;
static int twist_miss;
static TickType_t twist_guard_last;
static int move_speed = HAND_SPEED;
static int move_acc = HAND_ACC;
static int stream_ms;
static TickType_t stream_last;
static int cached_mode[ID_LAST + 1];
static uint8_t mode_ok[ID_LAST + 1];
static char ble_line[256];
static int ble_line_n;

static unsigned long now_ms(void)
{
    return (unsigned long)(xTaskGetTickCount() * portTICK_PERIOD_MS);
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
            if (ble_line_n > 0) {
                ble_nus_send(ble_line, ble_line_n);
                ble_nus_send("\n", 1);
                ble_line_n = 0;
            } else {
                ble_nus_send("\n", 1);
            }
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

static const char *id_name(int id)
{
    switch (id) {
    case 11: return "turn";
    case 12: return "raise-A";
    case 13: return "raise-B";
    case 14: return "bend";
    case 15: return "wrist";
    case 16: return "twist";
    case 17: return "hand";
    default: return "id";
    }
}

static const char *id_motor(int id)
{
    if (id == 17) {
        return "ST3215HS";
    }
    if (id == 16) {
        return "ST3215";
    }
    return "ST3235";
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
    printf("  get      pos/speed/load/V/temp/current/moving/torque-en + un/turns/st/mode\n");
    printf("  get N    same, one ID (16 = twist). faster than full get while moving\n");
    printf("  stream   live get all IDs every 180ms. stream 0/off stops. stream N = period ms\n");
    printf("  who      EEPROM: fw/mode/PID/angle-limit/goal per ID\n");
    printf("  who N    same, one ID\n");
    printf("  estop    ALL ids tq=0 + twist speed 0 + servo\n");
    printf("  batt     pack V from servo VIN + 3S rest-%% guess (not a fuel gauge)\n");
    printf("  bench    chip + ping 11-17\n");
    printf("  move speed N / acc N   next WritePos (default %d / %d)\n",
           HAND_SPEED, HAND_ACC);
    printf("  id N free|hold|to G|nudge D   any 11-17. 11-15 hop<=%d (no walls)\n",
           NUDGE_MAX);
    printf("  hand open   ID 17 to %d (soft open). slow. no other ID\n", HAND_OPEN);
    printf("  hand close  ID 17 to %d (soft close). slow. no other ID\n", HAND_CLOSE);
    printf("  hand to N   ID 17 to N, clamped %d-%d (wall 2088)\n", HAND_OPEN, HAND_TO_MAX);
    printf("  hand free   ID 17 torque off\n");
    printf("  twist to N  ID 16 to N, 0-4095, max hop %d (walk wraps)\n",
           TWIST_STEP_MAX);
    printf("  twist mid   ID 16 to %d (wrap-seam home, not unique)\n", TWIST_CENTER);
    printf("  twist free  ID 16 torque off\n");
    printf("  twist here  declare current pos as cable-home (host still saves un)\n");
    printf("  twist servo / wheel  mode 0 or 1 (EEPROM — always put back to servo)\n");
    printf("  twist speed N  wheel jog, signed. 0=stop. then twist servo\n");
    printf("  twist estop    same as estop for twist only\n");
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

static void joint_free(int id, const char *name)
{
    uint8_t tq = 0;
    if (!bus_write(id, REG_TORQUE_EN, &tq, 1)) {
        printf("%s free miss\n", name);
        return;
    }
    printf("%s free tq=0 id=%d\n", name, id);
}

static int read_pos(int id)
{
    uint8_t d[2];
    if (!bus_read(id, REG_PRESENT, 2, d)) {
        return -1;
    }
    return sts_s15((uint16_t)(d[0] | (d[1] << 8)));
}

static int twist_set_mode(uint8_t mode);

static int read_present(int id, present_t *p)
{
    uint8_t d[PRESENT_LEN];
    if (!bus_read(id, REG_PRESENT, PRESENT_LEN, d)) {
        return 0;
    }
    p->pos = sts_s15((uint16_t)(d[0] | (d[1] << 8)));
    p->speed = sts_s15((uint16_t)(d[2] | (d[3] << 8)));
    p->load = sts_s15((uint16_t)(d[4] | (d[5] << 8)));
    p->v_raw = d[6];
    p->temp = d[7];
    p->moving = d[10];
    p->cur = sts_s15((uint16_t)(d[13] | (d[14] << 8)));
    return 1;
}

static void twist_guard_stop(void)
{
    twist_guard_on = 0;
    twist_cruise_n = 0;
    twist_cruise_load = 0;
    twist_miss = 0;
}

static void twist_cut(const char *why, int load, int cur, int pos)
{
    uint8_t z[2] = { 0, 0 };
    uint8_t tq0 = 0;
    bus_write(TWIST_ID, REG_GOAL_SPEED, z, 2);
    bus_write(TWIST_ID, REG_TORQUE_EN, &tq0, 1);
    twist_set_mode(0);
    bus_write(TWIST_ID, REG_TORQUE_EN, &tq0, 1);
    twist_guard_stop();
    printf("ESTOP twist %s load=%d cur=%d pos=%d tq=0 servo\n",
           why, load, cur, pos);
}

static void twist_guard_start(void)
{
    twist_cruise_n = 0;
    twist_cruise_load = 0;
    twist_miss = 0;
    twist_guard_last = 0;
    twist_guard_on = 1;
}

static void twist_guard_tick(void)
{
    present_t p;
    TickType_t t;

    if (!twist_guard_on) {
        return;
    }
    t = xTaskGetTickCount();
    if (twist_guard_last != 0 &&
        (t - twist_guard_last) < pdMS_TO_TICKS(25)) {
        return;
    }
    twist_guard_last = t;
    if (!read_present(TWIST_ID, &p)) {
        twist_miss++;
        if (twist_miss >= GUARD_MISS) {
            twist_cut("bus-miss", 0, 0, -1);
        }
        return;
    }
    twist_miss = 0;
    if (p.speed > 40 || p.speed < -40) {
        if (twist_cruise_n < 4) {
            int al = p.load < 0 ? -p.load : p.load;
            twist_cruise_load = (twist_cruise_load * twist_cruise_n + al) /
                                (twist_cruise_n + 1);
            twist_cruise_n++;
            return;
        }
    }
    {
        int al = p.load < 0 ? -p.load : p.load;
        int ac = p.cur < 0 ? -p.cur : p.cur;
        if (twist_cruise_n >= 4 && al - twist_cruise_load >= GUARD_DLOAD) {
            twist_cut("dload", p.load, p.cur, p.pos);
            return;
        }
        if (al >= GUARD_LOAD) {
            twist_cut("load", p.load, p.cur, p.pos);
            return;
        }
        if (ac >= GUARD_CUR) {
            twist_cut("cur", p.load, p.cur, p.pos);
            return;
        }
    }
}

static int unwrap_pos(int id, int pos)
{
    if (pos_seen[id]) {
        int d = pos - last_pos[id];
        if (d > 2048) {
            wrap_turns[id]--;
        } else if (d < -2048) {
            wrap_turns[id]++;
        }
    }
    pos_seen[id] = 1;
    last_pos[id] = pos;
    return pos + wrap_turns[id] * 4096;
}

static void joint_to(int id, int goal, int lo, int hi, int hop_max, const char *name)
{
    if (goal < lo) {
        goal = lo;
    }
    if (goal > hi) {
        goal = hi;
    }
    int cur = read_pos(id);
    if (cur < 0) {
        printf("%s abort no pos id=%d\n", name, id);
        return;
    }
    unwrap_pos(id, cur);
    if (hop_max > 0) {
        int d = goal - cur;
        if (d > hop_max || d < -hop_max) {
            printf("%s abort hop %d cur=%d goal=%d max=%d\n",
                   name, d, cur, goal, hop_max);
            return;
        }
    }
    uint8_t v = 0;
    if (!bus_read(id, REG_VOLTAGE, 1, &v) || v < V_CUTOFF) {
        printf("%s abort pack v_raw=%d need>=%d\n", name, v, V_CUTOFF);
        return;
    }
    uint8_t tq = 1;
    if (!bus_write(id, REG_TORQUE_EN, &tq, 1)) {
        printf("%s torque miss\n", name);
        return;
    }
    uint8_t buf[7];
    buf[0] = (uint8_t)move_acc;
    buf[1] = (uint8_t)(goal & 0xFF);
    buf[2] = (uint8_t)((goal >> 8) & 0xFF);
    buf[3] = 0;
    buf[4] = 0;
    buf[5] = (uint8_t)(move_speed & 0xFF);
    buf[6] = (uint8_t)((move_speed >> 8) & 0xFF);
    if (!bus_write(id, REG_ACC, buf, 7)) {
        printf("%s write miss goal=%d\n", name, goal);
        return;
    }
    printf("%s to id=%d goal=%d speed=%d acc=%d clamp=%d-%d tq=1\n",
           name, id, goal, move_speed, move_acc, lo, hi);
}

static void clamps_for(int id, int *lo, int *hi, int *hop)
{
    *lo = 0;
    *hi = 4095;
    *hop = NUDGE_MAX;
    if (id == HAND_ID) {
        *lo = HAND_OPEN;
        *hi = HAND_TO_MAX;
        *hop = 0;
    } else if (id == TWIST_ID) {
        *hop = TWIST_STEP_MAX;
    }
}

static void cmd_hand(const char *what)
{
    if (strcmp(what, "free") == 0) {
        joint_free(HAND_ID, "hand");
        return;
    }
    int goal;
    int hi = HAND_CLOSE;
    if (strcmp(what, "open") == 0) {
        goal = HAND_OPEN;
    } else if (strcmp(what, "close") == 0) {
        goal = HAND_CLOSE;
    } else if (strncmp(what, "to ", 3) == 0) {
        goal = atoi(what + 3);
        hi = HAND_TO_MAX;
    } else {
        printf("err hand open|close|to N|free\n");
        return;
    }
    joint_to(HAND_ID, goal, HAND_OPEN, hi, 0, "hand");
}

static int twist_read_mode(void)
{
    uint8_t m = 0;
    if (!bus_read(TWIST_ID, REG_MODE, 1, &m)) {
        return -1;
    }
    cached_mode[TWIST_ID] = (int)m;
    mode_ok[TWIST_ID] = 1;
    return (int)m;
}

static int twist_set_mode(uint8_t mode)
{
    uint8_t tq = 0, unlock = 0, lock = 1;
    if (!bus_write(TWIST_ID, REG_TORQUE_EN, &tq, 1)) {
        return 0;
    }
    if (!bus_write(TWIST_ID, REG_LOCK, &unlock, 1)) {
        return 0;
    }
    if (!bus_write(TWIST_ID, REG_MODE, &mode, 1)) {
        bus_write(TWIST_ID, REG_LOCK, &lock, 1);
        return 0;
    }
    bus_write(TWIST_ID, REG_LOCK, &lock, 1);
    cached_mode[TWIST_ID] = (int)mode;
    mode_ok[TWIST_ID] = 1;
    return 1;
}

static void twist_write_speed(int speed)
{
    uint16_t u = (uint16_t)(speed < 0 ? -speed : speed);
    if (u > 1500) {
        u = 1500;
    }
    if (speed < 0) {
        u |= 0x8000;
    }
    uint8_t buf[2] = { (uint8_t)(u & 0xFF), (uint8_t)(u >> 8) };
    uint8_t tq = 1;
    bus_write(TWIST_ID, REG_TORQUE_EN, &tq, 1);
    if (!bus_write(TWIST_ID, REG_GOAL_SPEED, buf, 2)) {
        printf("twist speed miss\n");
        return;
    }
    printf("twist speed=%d tq=1\n", speed);
    if (speed == 0) {
        twist_guard_stop();
    } else {
        twist_guard_start();
    }
}

static void cmd_twist(const char *what)
{
    if (strcmp(what, "estop") == 0) {
        twist_cut("cmd", 0, 0, read_pos(TWIST_ID));
        return;
    }
    if (strcmp(what, "free") == 0) {
        twist_guard_stop();
        int mode = twist_read_mode();
        if (mode == 1) {
            twist_write_speed(0);
            twist_set_mode(0);
            printf("twist free restored servo mode\n");
        }
        joint_free(TWIST_ID, "twist");
        return;
    }
    if (strcmp(what, "here") == 0) {
        int pos = read_pos(TWIST_ID);
        printf("twist here pos=%d — host: this is cable-home turns=0. pack-off is stale\n", pos);
        return;
    }
    if (strcmp(what, "servo") == 0) {
        twist_guard_stop();
        twist_write_speed(0);
        if (!twist_set_mode(0)) {
            printf("twist servo miss\n");
            return;
        }
        printf("twist mode=0 servo\n");
        return;
    }
    if (strcmp(what, "wheel") == 0) {
        if (!twist_set_mode(1)) {
            printf("twist wheel miss\n");
            return;
        }
        printf("twist mode=1 wheel — jog with twist speed N, then twist servo\n");
        return;
    }
    if (strncmp(what, "speed ", 6) == 0) {
        int mode = twist_read_mode();
        if (mode != 1) {
            printf("twist speed abort mode=%d need wheel\n", mode);
            return;
        }
        twist_write_speed(atoi(what + 6));
        return;
    }
    int goal;
    if (strcmp(what, "mid") == 0) {
        goal = TWIST_CENTER;
    } else if (strncmp(what, "to ", 3) == 0) {
        goal = atoi(what + 3);
    } else {
        printf("err twist to N|mid|free|here|servo|wheel|speed N|estop\n");
        return;
    }
    if (twist_read_mode() != 0) {
        printf("twist abort not servo — twist servo first\n");
        return;
    }
    joint_to(TWIST_ID, goal, TWIST_MIN, TWIST_MAX, TWIST_STEP_MAX, "twist");
}

static void cmd_estop_all(void)
{
    uint8_t z[2] = { 0, 0 };
    uint8_t tq0 = 0;
    bus_write(TWIST_ID, REG_GOAL_SPEED, z, 2);
    twist_guard_stop();
    twist_set_mode(0);
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        bus_write(id, REG_TORQUE_EN, &tq0, 1);
    }
    printf("ESTOP all tq=0 twist servo\n");
}

static void cmd_id(const char *what)
{
    int id = atoi(what);
    const char *rest = what;
    const char *name;
    int lo, hi, hop;

    if (id < ID_FIRST || id > ID_LAST) {
        printf("err id 11-17\n");
        return;
    }
    while (*rest && !isspace((unsigned char)*rest)) {
        rest++;
    }
    while (*rest && isspace((unsigned char)*rest)) {
        rest++;
    }
    name = id_name(id);
    if (id == 12 || id == 13) {
        printf("warn raise is dual-drive 12+13 — one motor fights the other\n");
    }
    if (strcmp(rest, "free") == 0) {
        if (id == TWIST_ID) {
            twist_guard_stop();
        }
        joint_free(id, name);
        return;
    }
    if (strcmp(rest, "hold") == 0) {
        int pos = read_pos(id);
        if (pos < 0) {
            printf("%s abort no pos id=%d\n", name, id);
            return;
        }
        joint_to(id, pos, pos, pos, 0, name);
        return;
    }
    clamps_for(id, &lo, &hi, &hop);
    if (id == TWIST_ID &&
        (strncmp(rest, "to ", 3) == 0 || strncmp(rest, "nudge ", 6) == 0) &&
        twist_read_mode() != 0) {
        printf("twist abort not servo — twist servo first\n");
        return;
    }
    if (strncmp(rest, "to ", 3) == 0) {
        joint_to(id, atoi(rest + 3), lo, hi, hop, name);
        return;
    }
    if (strncmp(rest, "nudge ", 6) == 0) {
        int d = atoi(rest + 6);
        int pos;
        if (d > NUDGE_MAX) {
            d = NUDGE_MAX;
        }
        if (d < -NUDGE_MAX) {
            d = -NUDGE_MAX;
        }
        pos = read_pos(id);
        if (pos < 0) {
            printf("%s abort no pos id=%d\n", name, id);
            return;
        }
        joint_to(id, pos + d, lo, hi, NUDGE_MAX, name);
        return;
    }
    printf("err id N free|hold|to G|nudge D\n");
}

static void print_who_id(int id)
{
    uint8_t fw[2] = { 0, 0 };
    uint8_t ang[4] = { 0 };
    uint8_t pid[3] = { 0 };
    uint8_t tqlim[2] = { 0 };
    uint8_t goal[2] = { 0 };
    uint8_t mode = 0;
    int amin = -1, amax = -1, kp = -1, kd = -1, ki = -1, gl = -1, lim = -1;
    int md = -1;
    char fw_s[12];

    snprintf(fw_s, sizeof(fw_s), "?");
    if (bus_read(id, REG_FW, 2, fw)) {
        snprintf(fw_s, sizeof(fw_s), "%d.%d", fw[0], fw[1]);
    }
    if (bus_read(id, REG_MIN_ANGLE, 4, ang)) {
        amin = (int)(ang[0] | (ang[1] << 8));
        amax = (int)(ang[2] | (ang[3] << 8));
    }
    if (bus_read(id, REG_KP, 3, pid)) {
        kp = pid[0];
        kd = pid[1];
        ki = pid[2];
    }
    if (bus_read(id, REG_MAX_TORQUE, 2, tqlim)) {
        lim = (int)(tqlim[0] | (tqlim[1] << 8));
    }
    if (bus_read(id, REG_MODE, 1, &mode)) {
        md = (int)mode;
        cached_mode[id] = md;
        mode_ok[id] = 1;
    }
    if (bus_read(id, REG_GOAL, 2, goal)) {
        gl = sts_s15((uint16_t)(goal[0] | (goal[1] << 8)));
    }
    printf("who id=%d name=%s fw=%s mode=%d kp=%d kd=%d ki=%d amin=%d amax=%d goal=%d tqlim=%d motor=%s\n",
           id, id_name(id), fw_s, md, kp, kd, ki, amin, amax, gl, lim, id_motor(id));
}

static void cmd_who(int id)
{
    if (id > 0) {
        if (id < ID_FIRST || id > ID_LAST) {
            printf("err who 11-17\n");
            return;
        }
        print_who_id(id);
        return;
    }
    printf("who EEPROM+RAM. mode 0=servo 1=wheel. amin/amax=0-4095 means no eeprom wall.\n");
    for (int i = ID_FIRST; i <= ID_LAST; i++) {
        print_who_id(i);
    }
}

static void cmd_move(const char *what)
{
    if (strncmp(what, "speed ", 6) == 0) {
        int s = atoi(what + 6);
        if (s < 1) {
            s = 1;
        }
        if (s > 3073) {
            s = 3073;
        }
        move_speed = s;
        printf("move speed=%d\n", move_speed);
        return;
    }
    if (strncmp(what, "acc ", 4) == 0) {
        int a = atoi(what + 4);
        if (a < 0) {
            a = 0;
        }
        if (a > 150) {
            a = 150;
        }
        move_acc = a;
        printf("move acc=%d\n", move_acc);
        return;
    }
    printf("err move speed N|acc N  now speed=%d acc=%d\n", move_speed, move_acc);
}

static void cmd_stream(const char *what)
{
    while (*what && isspace((unsigned char)*what)) {
        what++;
    }
    if (*what == '\0' || strcmp(what, "on") == 0) {
        stream_ms = 180;
    } else if (strcmp(what, "off") == 0) {
        stream_ms = 0;
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
        stream_ms = ms;
    }
    if (stream_ms) {
        listen_dump = 0;
    }
    stream_last = 0;
    printf("stream ms=%d dump=%d\n", stream_ms, listen_dump);
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
    printf("info idf=%s usb=115200 bus=gpio18/19@1Mbps ids=11-17 ble=%s\n",
           esp_get_idf_version(), ble_nus_name());
    printf("info note=no auto-move. same text on USB and BLE. servos need XT60\n");
}

static void cmd_state(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("state t=%lu heap=%u reset=%s tasks=%u bus_rx=%lu dump=%d stream=%d ble=%d\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           reset_str(esp_reset_reason()),
           (unsigned)uxTaskGetNumberOfTasks(),
           (unsigned long)bus_rx_total,
           listen_dump, stream_ms, ble_nus_connected());
    printf("state mac=%02x:%02x:%02x:%02x:%02x:%02x name=%s\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], ble_nus_name());
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
    printf("bus=gpio18/19@1Mbps ids=%d-%d ble=%s\n", ID_FIRST, ID_LAST, ble_nus_name());
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

static void print_get_id(int id)
{
    uint8_t d[PRESENT_LEN];
    uint8_t tq = 0;
    char mode_s[8];

    if (!bus_read(id, REG_PRESENT, PRESENT_LEN, d)) {
        printf("get id=%d miss\n", id);
        return;
    }
    int pos = sts_s15((uint16_t)(d[0] | (d[1] << 8)));
    int speed = sts_s15((uint16_t)(d[2] | (d[3] << 8)));
    int load = sts_s15((uint16_t)(d[4] | (d[5] << 8)));
    int v_raw = d[6];
    int temp = d[7];
    int st = d[9];
    int moving = d[10];
    int current = sts_s15((uint16_t)(d[13] | (d[14] << 8)));
    int tq_ok = bus_read(id, REG_TORQUE_EN, 1, &tq);
    int un = unwrap_pos(id, pos);
    if (mode_ok[id]) {
        snprintf(mode_s, sizeof(mode_s), "%d", cached_mode[id]);
    } else {
        snprintf(mode_s, sizeof(mode_s), "?");
    }
    if (load > 400 || load < -400) {
        printf("get LOAD id=%d load=%d pos=%d un=%d\n", id, load, pos, un);
    }
    printf("get id=%d pos=%d speed=%d load=%d v_raw=%d v=%.1f temp=%d cur=%d moving=%d tq=%s un=%d turns=%d st=%d mode=%s\n",
           id, pos, speed, load, v_raw, v_raw / 10.0, temp, current, moving,
           tq_ok ? (tq ? "1" : "0") : "?", un, wrap_turns[id], st, mode_s);
}

static void cmd_get(void)
{
    printf("get present+torque-en. writes: id N / hand / twist\n");
    twist_guard_tick();
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        print_get_id(id);
        twist_guard_tick();
    }
}

static void cmd_get_one(int id)
{
    if (id < 1 || id > 253) {
        printf("err get id 1-253\n");
        return;
    }
    twist_guard_tick();
    print_get_id(id);
}

static void stream_tick(void)
{
    TickType_t t;

    if (stream_ms <= 0) {
        return;
    }
    t = xTaskGetTickCount();
    if (stream_last != 0 && (t - stream_last) < pdMS_TO_TICKS(stream_ms)) {
        return;
    }
    stream_last = t;
    twist_guard_tick();
    for (int id = ID_FIRST; id <= ID_LAST; id++) {
        print_get_id(id);
        twist_guard_tick();
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
    printf("ble=%s nordic-uart. same text as USB.\n", ble_nus_name());
    printf("note=no auto-move. hand 689-2075. twist hop<=900. id 11-15 hop<=%d.\n",
           NUDGE_MAX);
}

static void heartbeat(void)
{
    TickType_t t = xTaskGetTickCount();
    if ((t - last_beat) < pdMS_TO_TICKS(2000)) {
        return;
    }
    last_beat = t;
    printf("t=%lu heap=%u bus_rx=%lu ble=%d stream=%d\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           (unsigned long)bus_rx_total,
           ble_nus_connected(), stream_ms);
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
    if (strncmp(line, "get ", 4) == 0) {
        cmd_get_one(atoi(line + 4));
        return;
    }
    if (strcmp(line, "who") == 0) {
        cmd_who(0);
        return;
    }
    if (strncmp(line, "who ", 4) == 0) {
        cmd_who(atoi(line + 4));
        return;
    }
    if (strcmp(line, "stream") == 0) {
        cmd_stream("");
        return;
    }
    if (strncmp(line, "stream ", 7) == 0) {
        cmd_stream(line + 7);
        return;
    }
    if (strncmp(line, "move ", 5) == 0) {
        cmd_move(line + 5);
        return;
    }
    if (strncmp(line, "id ", 3) == 0) {
        cmd_id(line + 3);
        return;
    }
    if (strcmp(line, "estop") == 0) {
        cmd_estop_all();
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
    if (strncmp(line, "twist ", 6) == 0) {
        cmd_twist(line + 6);
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
    ble_nus_init();

    print_banner();
    print_help();
    cmd_state();

    char line[64];
    int line_len = 0;
    for (;;) {
        char bline[64];

        bus_eat(1);
        heartbeat();
        twist_guard_tick();
        stream_tick();
        ble_nus_tick();
        if (ble_nus_subscribed() && listen_dump) {
            listen_dump = 0;
        }
        while (ble_nus_pop_line(bline, (int)sizeof(bline))) {
            handle_line(bline);
        }

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
