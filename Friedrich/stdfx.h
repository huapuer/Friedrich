#pragma once
#define ERROR(format,...) do{fprintf(stderr,format,##__VA_ARGS__);system("pause");exit(1);}while(0)