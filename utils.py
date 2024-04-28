def print_bold(*msgs):
    bold_msgs = ['\033[1m' + str(msg) + '\033[0m' for msg in msgs]
    print('>>', *bold_msgs)

def print_bold_vars(*msgs):
    bold_msgs = []
    for msg in msgs:
        if isinstance(msg, str):
            bold_msgs.append(msg)
        elif isinstance(msg, float):
            bold_msgs.append('\033[1m' + str(round(msg, 2)) + '\033[0m')
        else:
            bold_msgs.append('\033[1m' + str(msg) + '\033[0m')
    print('>>', *bold_msgs)

def print_color(*msgs, color='green'):
    if color == 'red':
        colored_msgs = ['\033[91m' + str(msg) + '\033[0m' for msg in msgs]
    elif color == 'green':
        colored_msgs = ['\033[92m' + str(msg) + '\033[0m' for msg in msgs]
    elif color == 'yellow':
        colored_msgs = ['\033[93m' + str(msg) + '\033[0m' for msg in msgs]
    else:
        colored_msgs = msgs
    print('>>', *colored_msgs)