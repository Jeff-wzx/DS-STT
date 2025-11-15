import configargparse as argparse
import re
from collections import OrderedDict

# 类型转换器。此函数将字符串值转换为布尔值
def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 参数的类型转换器。此函数将表示元组的字符串值转换为元组数字列表
# 正则表达式（Regular Expression)是一种文本模式
# 作用是将正则表达式的字符串形式编译为一个正则表达式对象，这样可以提高正则匹配的效率。
# 使用 re.compile() 后，可以使用该对象的方法进行匹配和替换操作。
# 正则表达式用于匹配文本中的十进制数或整数。
# \d+：此部分匹配一个或多个数字；\.：这匹配文字点字符（使用反斜杠转义）；\d+：这将匹配点后的一个或多个数字。|：是管道符号，充当 OR 运算符
# 从字符串中解析元组、元组列表和字典，并将提取的值转换为函数定义的特定类型

# 以匹配十进制或整数
def arg_tuple(cast_type):

    regex = re.compile(r'\d+\.\d+|\d+')

    def parse_tuple(v):
        vals = regex.findall(v)
        return [cast_type(val) for val in vals]

    return parse_tuple

# 匹配括在括号中的元组。
def arg_list_tuple(cast_type):
    regex = re.compile(r'\([^\)]*\)')
    tuple_parser = arg_tuple(cast_type)

    def parse_list(v):
        tuples = regex.findall(v)
        return [tuple_parser(t) for t in tuples]

    return parse_list


def arg_dict(cast_type):
    regex_pairs = re.compile(r'[^\ ]+=[^\ ]+')
    regex_keyvals = re.compile(r'([^\ ]+)=([^\ ]+)')

    def parse_dict(v):
        d = OrderedDict()
        for keyval in regex_pairs.findall(v):
            key, val = regex_keyvals.match(keyval).groups()
            d.update({key:cast_type(val)})
        return d

    return parse_dict
