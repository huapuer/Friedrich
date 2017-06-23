#pragma once
#include <stdio.h>
#include "../../Ludwig/Ludwig/ludwig_net.h"

void acts_test(char* c, int size) {
	for (int i = 0; i < size; i++) {
		printf("%c", c[i]);
	}
	printf("\n");
	//friedrich_says(net_events::EVENT_TEST, "How are you Alan.", 17);
}