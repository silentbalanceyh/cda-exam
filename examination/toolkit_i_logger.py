# ========================= 日志函数
def __log(content, arg1=None):
    if arg1 is None:
        print(content)
    else:
        print(content, arg1)

def log_info(content, arg1=None):
    __log("\033[34m[I] %s" % content, arg1)


def log_message(content, arg1=None):
    __log("\033[37m[I] %s " % content, arg1)

def log_matrix(content, x, y=None):
    if y is None:
        literal = "\033[33mX = %s\033[37m" % (x,)
    else:
        literal = "\033[33mX = %s, Y = %s\033[37m" % (x, y)
    __log("\033[37m[I] （Matrix）%s " % content, literal)

def log_success(content, arg1=None):
    __log("\033[32m[I] %s " % content, arg1)


def log_warn(content, arg1=None):
    __log("\033[33m[W] %s" % content, arg1)


def log_error(content, arg1=None):
    __log("\033[31m[E] %s" % content, arg1)