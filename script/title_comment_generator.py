#!/usr/bin/python
"""
Title Comment Generator
=======================

Generate title kind of comments on 80 columns, like this:

##################################### TEST #####################################

:Example:
    $ python title_comment_generator.py -c TEST

Use '-c' to copy the title automatically to the clipboard.
"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="The title you wan't to put in comment")
    parser.add_argument("-c", "--copy", action="store_true", default=False, help="Copy to clipboard")
    args = vars(parser.parse_args())
    title = " " + str(args['title']) + " "
    title = title.center(80, '#')
    print title
    if args['copy']:
        import os
        os.system("echo '%s' | pbcopy" % title)
