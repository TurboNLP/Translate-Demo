#!/usr/bin/env python
# coding: utf-8

info = '''
Command: subEntity.py input output
Purpose: Do the following substitutions:
         1, Character Entities ==> Value
         2, _U_._S_._          ==> _U.S._
Author:  Meng Fandong
'''

import re

entity_list = ['&amp;', '& amp;', '&amp ;', '& amp ;',
               '&AMP;', '& AMP;', '&AMP ;', '& AMP ;',

               '&lt;', '& lt;', '&lt ;', '& lt ;',
               '&LT;', '& LT;', '&LT ;', '& LT ;',

               '&gt;', '& gt;', '&gt ;', '& gt ;',
               '&GT;', '& GT;', '&GT ;', '& GT ;',

               '&apos;', '& apos;', '&apos ;', '& apos ;',
               '&APOS;', '& APOS;', '&APOS ;', '& APOS ;',

               '&quot;', '& quot;', '&quot ;', '& quot ;',
               '&QUOT;', '& QUOT;', '&QUOT ;', '& QUOT ;'
               ]

value_list = ['&', '<', '>', '\'', '\"']


def E2V(sequence):
    "Substitute the Value for the Character Entities in the file"
    for i in range(len(entity_list)):
        (sequence, times) = re.subn(entity_list[i], value_list[int(i / 8)], sequence)
    (sequence, times) = re.subn(' U . S . ', ' U.S. ', sequence)
    return sequence
