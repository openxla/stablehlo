#include <signal.h>
#include <unistd.h>

int x() {
    return 1;
}

int y() {
    return 1 + x();
}

int z() {
    return 2 + y();
}

int main() {
    kill(getpid(), SIGABRT);
    return 3 + z();
} 
