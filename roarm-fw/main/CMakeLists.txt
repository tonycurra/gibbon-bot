idf_component_register(SRCS "main.c" "ble_nus.c" INCLUDE_DIRS "."
                       REQUIRES driver vfs esp_driver_uart esp_hw_support spi_flash bt nvs_flash)
