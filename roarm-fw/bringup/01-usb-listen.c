#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_idf_version.h"
#include "esp_mac.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define FW "01-usb-listen"
#define USB_UART UART_NUM_0

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
           (unsigned long)(flash_size / 1024),
           reset_str(esp_reset_reason()));
    printf("info mac=%02x:%02x:%02x:%02x:%02x:%02x heap=%u uptime_ms=%lu\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
           (unsigned)esp_get_free_heap_size(), now_ms());
    printf("info idf=%s usb=115200 fw=%s ble=0 bus=0\n",
           esp_get_idf_version(), FW);
    printf("info note=usb only. no move. pack may be off.\n");
}

static void cmd_state(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("state t=%lu heap=%u reset=%s tasks=%u\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           reset_str(esp_reset_reason()),
           (unsigned)uxTaskGetNumberOfTasks());
    printf("state mac=%02x:%02x:%02x:%02x:%02x:%02x fw=%s\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], FW);
}

static void print_banner(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    printf("\nbringup %s\n", FW);
    printf("mac=%02x:%02x:%02x:%02x:%02x:%02x\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("reset=%s\n", reset_str(esp_reset_reason()));
    printf("tasks=%u heap=%u\n",
           (unsigned)uxTaskGetNumberOfTasks(),
           (unsigned)esp_get_free_heap_size());
    printf("note=usb only. no ble. no bus. no move.\n");
}

static void heartbeat(void)
{
    TickType_t t = xTaskGetTickCount();
    if ((t - last_beat) < pdMS_TO_TICKS(2000)) {
        return;
    }
    last_beat = t;
    printf("t=%lu heap=%u reset=%s\n",
           now_ms(),
           (unsigned)esp_get_free_heap_size(),
           reset_str(esp_reset_reason()));
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
        printf("commands: help info state\n");
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

    last_beat = xTaskGetTickCount();
    print_banner();
    cmd_state();

    char line[64];
    int line_len = 0;
    for (;;) {
        heartbeat();
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
