#pragma once
#include <stdio.h>
#include "../../Ludwig/Ludwig/ludwig_net.h"

void acts_test(char* c, int size) {
	printf("From Alan£º%s \r\n", c);
	friedrich_says(net_events::EVENT_TEST, "How are you Alan.", 17);
}