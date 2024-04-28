def print_bold(*msgs):
    bold_msgs = ['\033[1m' + msg + '\033[0m' for msg in msgs]
    print('>>', *bold_msgs)