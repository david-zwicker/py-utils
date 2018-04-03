#!/usr/bin/env python2

"""
Python function for processing bibtex files.

This code and information is provided 'as is' without warranty of any kind,
either express or implied, including, but not limited to, the implied
warranties of non-infringement, merchantability or fitness for a particular
purpose.
"""

import os, sys
import re
import itertools
import codecs
import logging

from optparse import OptionParser
from pybtex.database.input import bibtex

########################
# DEFAULTS
########################

#remove_items = set(['URL', 'doi'])

# papers has some weird classes => translate them to more common bibtex ones
class_translator = {
    'book': 'book',
    'article': 'article',
    'letter': 'article',
    'review': 'article',
    'editorialmaterial': 'article',
    'note': 'article',
    'newsitem': 'article',
    'proceedingspaper': 'inproceedings',
}
# allowed item per class
fields = {
    'article': set(['title', 'year', 'journal', 'pages', 'volume']),
    'inproceedings': set(['title', 'year', 'booktitle']),
    'book': set(['title', 'year', 'publisher']),
}
# general things to replace to become more LaTeX compatible
general_replacements = {
    u'\u0263': u'$\\gamma$',
    u'\u03B3': u'$\\gamma$',
    u'{gamma}': u'$\\gamma$',
    u'1st': u'\\nth{1}',
    u'2nd': u'\\nth{2}',
    u'3rd': u'\\nth{3}',
    u'C. elegans': u'{C.~elegans}',
    u'RNA': u'{RNA}',
}
# Journals where it's ok to have only one page cited
onepage_journals = set([None,
    "Phys. Rev. Lett.", "Phys. Rev. E", "New Journal of Physics", "PLOS ONE",
    "Nat. Commun.", "J. Stat. Mech.", "Open Biology", "BMC Biol."
])
# journals where the volume may be omitted
novolume_journals = set([None,
    "J. Stat. Mech."
])
# specific style definitions
styles = {
    'default': {'namedot': True},
    'pnas': {'namedot': False},
}

########################
# PARSE ARGUMENT LIST
########################

parser = OptionParser("%s [options] input [input]..." % sys.argv[0])
parser.add_option(
    "-o", "--output", dest="output", help="write resulting database to FILE",
    metavar="FILE"
)
parser.add_option(
    "-i", "--ignore", dest="ignore_files", action="append", default=[],
    help="entries from this file are ignored", metavar="FILE"
)
parser.add_option(
    "-s", "--style", dest="style", help="choose a style definitions"
)

# parse options from the command line
(options, args) = parser.parse_args()

# output help, if no arguments are given
if len(args) == 0:
    parser.print_help()
    exit()

# choose the right style
try:
    style = styles[options.style.lower()]
except (AttributeError, KeyError):
    style = styles['default']

########################
# PARSE BIBTEX
########################

def format_person(person, namedot=True):
    """ Formats a single name given in person """
    result = []
    for name in itertools.chain(person.first(), person.middle()):
        result.append(name[0])

    if namedot:
        result.append(' '.join(person.last()))
        return '. '.join(result)
    else:
        return ''.join(result) + ' ' + ' '.join(person.last())


def stringtype(s):
    """ displays the type of string `s` """
    if isinstance(s, str):
        print("ordinary string")
    elif isinstance(s, unicode):
        print("unicode string")
    else:
        print("not a string")

bib_entry = u"""
@{type}{{{key},
author = {{{author}}},
{fields}
}}
"""

# get those entries which should be ignored
ignore_keys = set()
for ignore_file in options.ignore_files:
    bib_ignore = bibtex.Parser().parse_file(os.path.expanduser(ignore_file))
    ignore_keys.update(bib_ignore.entries.keys())

# open output file if given, otherwise use stdout
if options.output:
    output = codecs.open(
        os.path.expanduser(options.output), mode='w', encoding='utf-8'
    )
else:
    output = os.fdopen(sys.stdout.fileno(), 'w', 0)

# loop over input files
for in_file in args:

    # load data
    bib_data = bibtex.Parser().parse_file(in_file)
    re_pages = re.compile(
        r'^\s*([a-zA-Z]?(\d+))\s*(\-+\s*([a-zA-Z]?(\d+))\s*)?$'
    )
    keys_seen = set()

    # iterate over all keys
    for key in bib_data.entries.keys():

        # ignore keys with colon or if they are in ignore_keys
        if ':' in key or key in ignore_keys:
            continue

        # check for duplicated keys
        if key in keys_seen:
            logging.warning('Key "%s" has already been used.', key)
        keys_seen.add(key)

        item = bib_data.entries[key]
        try:
            item_type = class_translator[item.type.lower()]
        except KeyError:
            logging.warning(
                'Item type "%s" unknown. Using "article" instead.',
                item.type.lower()
            )
            item_type = 'article'

        # format authors
        try:
            item_author = ' and '.join(
                format_person(p, style['namedot']) for p in item.persons['author']
            )
        except KeyError:
            logging.warning(
                '%s "%s" from "%s" has no author assigned',
                item_type, key, item.fields.get('journal')
            )

        # format pages
        if 'pages' in fields[item_type] and item.fields.has_key('pages'):

            # read pages
            match = re_pages.match(item.fields['pages'])
            try:
                first = match.group(1)
                first_int = int(match.group(2))
            except AttributeError:
                logging.warning(
                    '%s "%s" has strange pages `%s`',
                    item_type, key, item.fields['pages']
                )
            else:
                last = match.group(4)
                last_int = 1e10 if last is None else int(match.group(5))

                # check whether last page exists
                if last is None:
                    if item.fields.get('journal') not in onepage_journals:
                        logging.warning(
                            '%s "%s" from "%s" has only a single page assigned',
                            item_type, key, item.fields.get('journal')
                        )

                else:
                    if len(first) > len(last):
                        last = first[:len(first)-len(last)] + last

                    elif first_int > last_int:
                        logging.warning(
                            '%s "%s" has page %s..%s',
                            item_type, key, first, last
                        )

                    item.fields['pages'] = '%s-%s' % (first, last)

        # format remaining fields
        item_fields = []
        for k in fields[item_type]:
            try:
                value = item.fields[k]
                for v1, v2 in general_replacements.iteritems():
                    value = value.replace(v1, v2)
                if k == 'title':
                    value = "{%s}" % value
                item_fields.append('%s = {%s},' % (k, value))
            except KeyError:
                if k != 'volume' or \
                        item.fields.get('journal') not in novolume_journals:

                    logging.warning(
                        'key "%s" does not exists for %s "%s"',
                        k, item_type, key
                    )

        # produce output
        entry = bib_entry.format(
            type=item_type,
            key=item.key,
            author=item_author,
            fields="\n".join(item_fields)
        )

        output.writelines(entry)
