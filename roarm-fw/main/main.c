#include <stdio.h>
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
    printf("\nroarm-fw usb hello\n");

    esp_chip_info_t chip;
    esp_chip_info(&chip);

    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);

    printf("chip: ESP32 cores=%d%s%s rev=%d flash=%luKB\n",
           chip.cores,
           (chip.features & CHIP_FEATURE_WIFI_BGN) ? " wifi" : "",
           (chip.features & CHIP_FEATURE_BT) ? " bt" : "",
           chip.revision,
           (unsigned long)(flash_size / 1024));
    printf("usb serial 115200 — servo bus not started\n");

    for (;;) {
        printf("alive\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
