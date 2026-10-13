set(NUS "${CMAKE_CURRENT_LIST_DIR}/../../main/ble_nus.c")
idf_component_register(SRCS "app.c" "${NUS}"
                       INCLUDE_DIRS "." "${CMAKE_CURRENT_LIST_DIR}/../../main"
                       REQUIRES driver vfs esp_driver_uart esp_hw_support spi_flash bt nvs_flash)
target_compile_definitions(${COMPONENT_LIB} PRIVATE GIBBON_MONITOR=1)
