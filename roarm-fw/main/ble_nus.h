#ifndef BLE_NUS_H
#define BLE_NUS_H

#include <stddef.h>

/* Nordic UART Service. Same text lines as USB. */

void ble_nus_init(void);
int ble_nus_connected(void);
int ble_nus_subscribed(void);
void ble_nus_tick(void);
const char *ble_nus_name(void);
void ble_nus_send(const void *data, int n);
/* 1 = a complete line (no newline) was copied into out. */
int ble_nus_pop_line(char *out, int max);

#endif
