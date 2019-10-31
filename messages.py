from colorama import init
from colorama import Back
from colorama import Fore
from colorama import Style

init()


def newline(n=1):
    for i in range(n):
        print("\r       ")


def colorize(msg, color):
    return "%s%s%s" % (color, msg, Style.RESET_ALL)


def info(msg):
    color = Fore.GREEN
    print("\r%s[i] %s" % (color, msg))


def normal(msg):
    color = Style.RESET_ALL
    print("\r%s[~] %s" % (color, msg))


def alert(msg):
    color = Fore.YELLOW
    print("\r%s[*] %s" % (color, msg))


def error(msg, stop=True):
    color = Back.RED + Fore.BLACK
    print("\r%s[!] %s" % (color, msg))

    if stop:
        exit()


def confirm(msg):
    color = Fore.YELLOW

    while True:
        r = raw_input("\r%s[?] %s (y/n) " % (color, msg))

        if r == 'Y' or r == 'y':
            return True
        elif r == 'N' or r == 'n':
            return False
        else:
            print("\r[!] please insert Y or N")


def confirm_enter(msg):
    color = Style.RESET_ALL
    raw_input("\r%s[i] %s, press enter to continue..." %(color, msg))
