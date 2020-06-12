#include "compat_switch.h"

#include <stdio.h>
#include <string.h>

#include "../Platform.h"

namespace Platform
{

void StopEmu()
{
}

FILE* OpenFile(const char* path, const char* mode, bool mustexist)
{
    FILE* ret;

    if (mustexist)
    {
        ret = fopen(path, "rb");
        if (ret) ret = freopen(path, mode, ret);
    }
    else
        ret = fopen(path, mode);

    return ret;
}


FILE* OpenLocalFile(const char* path, const char* mode)
{
    FILE* ret = fopen(path, mode);
    if (ret)
        return ret;

    const char* configDir = "/melonds/";
    int configPathLen = strlen(configDir);
    int pathLen = strlen(path);

    char* resPath = new char[configPathLen + pathLen + 1];
    strcpy(resPath, configDir);
    strcpy(resPath + configPathLen, path);

    ret = fopen(resPath, mode);
    delete[] resPath;

    return ret;
}

FILE* OpenDataFile(const char* path)
{
	return OpenLocalFile(path, "rb");
}

void ThreadEntry(void* param)
{
    printf("thread activated\n");
    ((void (*)())param)();
}

#define STACK_SIZE (1024 * 64)

int threadNextCore = 1;

void* Thread_Create(void (*func)())
{
    Thread* thread = new Thread();
    auto res = threadCreate(thread, ThreadEntry, (void*)func, NULL, STACK_SIZE, 0x30, threadNextCore++);
    threadStart(thread);
    printf("%d thread\n", res);
    return (void*)thread;
}

void Thread_Free(void* thread)
{
    threadClose((Thread*)thread);
    delete ((Thread*)thread);
}

void Thread_Wait(void* thread)
{
    threadWaitForExit((Thread*)thread);
}

void* Semaphore_Create()
{
    Semaphore* sema = new Semaphore();
    semaphoreInit(sema, 0);
    return (void*)sema;
}

void Semaphore_Free(void* sema)
{
    delete (Semaphore*)sema;
}

void Semaphore_Reset(void* sema)
{
    while(semaphoreTryWait((Semaphore*)sema));
}

void Semaphore_Wait(void* sema)
{
    semaphoreWait((Semaphore*)sema);
}

void Semaphore_Post(void* sema)
{
    semaphoreSignal((Semaphore*)sema);
}

void* GL_GetProcAddress(const char* proc)
{
    return NULL;
}

bool MP_Init()
{
    return false;
}

void MP_DeInit()
{}

int MP_SendPacket(u8* data, int len)
{
    return 0;
}

int MP_RecvPacket(u8* data, bool block)
{
    return 0;
}

bool LAN_Init()
{
    return false;
}

void LAN_DeInit()
{}

int LAN_SendPacket(u8* data, int len)
{
    return 0;
}

int LAN_RecvPacket(u8* data)
{
    return 0;
}

}