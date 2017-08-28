#pragma once
#define ERROR(format,...) do{fprintf(stderr,format,##__VA_ARGS__);system("pause");exit(1);}while(0)

#define DEBUG_SCHEDULE
//#define DEBUG_DATA
//#define DEBUG_EXTERNAL
//#define DEBUG_INIT_LAYER
//#define DEBUG_INIT_LINK
//#define DEBUG_KERNEL_PUSH_FULL
#define DEBUG_KERNEL_INTEGRATE
//#define DEBUG_KERNEL_CLEAR_PULL_FULL