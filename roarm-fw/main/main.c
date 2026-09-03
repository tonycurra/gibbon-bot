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

static void print_help(void)
{
    printf("commands:\n");
    printf("  info     ESP32 / USB (works with battery unplugged)\n");
    printf("  ping     Feetech ping IDs 11-17 (needs servo power + UART switch ESP32)\n");
    printf("  pingall  ping IDs 1-253 (slow)\n");
    printf("  help\n");
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
           (unsigned)esp_get_free_heap_size(),
           (unsigned long)(xTaskGetTickCount() * portTICK_PERIOD_MS));
    printf("info idf=%s usb=115200 bus=gpio18/19@1Mbps ids=11-17\n",
           esp_get_idf_version());
    printf("info note=servos need XT60; ping without battery is silence\n");
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

static int bus_ping(int id)
{
    uint8_t pkt[6];
    pkt[0] = 0xFF;
    pkt[1] = 0xFF;
    pkt[2] = (uint8_t)id;
    pkt[3] = 2;
    pkt[4] = 1; /* PING */
    pkt[5] = (uint8_t)(~(id + 2 + 1));

    uart_flush(BUS_UART);
    uart_write_bytes(BUS_UART, (const char *)pkt, sizeof(pkt));
    uart_wait_tx_done(BUS_UART, pdMS_TO_TICKS(5));

    uint8_t rx[16];
    int n = uart_read_bytes(BUS_UART, rx, sizeof(rx), pdMS_TO_TICKS(8));
    if (n < 6) {
        return 0;
    }
    /* skip possible echo of our ping */
    int i = 0;
    if (n >= 6 && rx[0] == 0xFF && rx[1] == 0xFF && rx[2] == id && rx[4] == 1) {
        i = 6;
    }
    if (n - i < 6) {
        return 0;
    }
    return (rx[i] == 0xFF && rx[i + 1] == 0xFF);
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
    if (strcmp(line, "ping") == 0) {
        cmd_ping(ID_FIRST, ID_LAST, 1);
        return;
    }
    if (strcmp(line, "pingall") == 0) {
        cmd_ping(1, 253, 0);
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

    printf("\nroarm-fw usb+bus\n");
    print_help();
    cmd_info();
    printf("type ping  (expect silence until battery is on)\n");

    char line[64];
    int line_len = 0;
    for (;;) {
        uint8_t ch;
        int n = uart_read_bytes(USB_UART, &ch, 1, pdMS_TO_TICKS(50));
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
