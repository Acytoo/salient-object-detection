#ifdef _WIN32 || _WIN64
#include <direct.h>
#include <io.h>
#elif __LINUX__
#include <stdarg.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32 || _WIN64
#define ACCESS _access
#define MKDIR(a) _mkdir((a))
#elif __LINUX__
#define ACCESS access
#define MKDIR(a) mkdir((a),0755)
#endif
