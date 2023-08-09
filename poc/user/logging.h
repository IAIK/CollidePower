#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

inline void print_system_config() {
    int r = 0;
    r += std::system("sudo date | sed 's/^/# /g' 1>&2");
    r += std::system("sudo pwd | sed 's/^/# /g' 1>&2");
    r += std::system("ls *.cpp | xargs -i sh -c \"echo {}; cat {};\" | sed 's/^/# /g' 1>&2");
    r += std::system("ls *.h | xargs -i sh -c \"echo {}; cat {};\" | sed 's/^/# /g' 1>&2");
    r += std::system("ls ../module/*.c | xargs -i sh -c \"echo {}; cat {};\" | sed 's/^/# /g' 1>&2");
    r += std::system("ls ../module/*.h | xargs -i sh -c \"echo {}; cat {};\" | sed 's/^/# /g' 1>&2");

    r += std::system("sudo cat /proc/cmdline | sed 's/^/# /g' 1>&2");
    r += std::system("sudo uname -r 2>/dev/null | sed 's/^/# /g' 1>&2");
    r += std::system("sudo lsb_release -a  2>/dev/null | sed 's/^/# /g' 1>&2");

    r += std::system("sudo cat /proc/cpuinfo | sed 's/^/# /g' 1>&2");
    r += std::system("sudo cpupower frequency-info 2>/dev/null | sed 's/^/# /g' 1>&2");
    r += std::system("sudo undervolt -r  2>/dev/null | sed 's/^/# /g' 1>&2");

    r += std::system("sudo dmesg | tail | sed 's/^/# /g' 1>&2");

    r += std::system("sudo cpuid -1 | sed 's/^/# /g' 1>&2");

    (void)r;
}

// https://stackoverflow.com/questions/9596945/how-to-get-appropriate-timestamp-in-c-for-logs
inline void timestamp() {
    time_t ltime = time(NULL);
    // asctime is super hacky ... it returns a pointer to a global char array
    // so no free but not thread-safe ... but logging from multiple threads breaks it anyways
    char *str = asctime(localtime(&ltime));

    str[strlen(str) - 1] = 0;
    fprintf(stderr, "%s", str);
}