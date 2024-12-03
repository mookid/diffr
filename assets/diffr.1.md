---
title: DIFFR
section: 1
header: User Manual
footer: diffr 1.5.0
date: April 14, 2023
---
# NAME
diffr - adds word-level diff on top of unified diffs

# SYNOPSIS
**diffr** [**\-\-colors** *\<color_spec\>*] [**\-\-line-numbers** \<compact|aligned\>]

diff -u <file1> <file2> | **diffr** [OPTIONS]

git show | **diffr** [OPTIONS]

# DESCRIPTION
**\-\-colors** *\<color_spec\>*
    Configure color settings for console ouput.

    There are four faces to customize:
    +----------------+--------------+----------------+
    |  line prefix   |      +       |       -        |
    +----------------+--------------+----------------+
    | common segment |    added     |    removed     |
    | unique segment | refine-added | refine-removed |
    +----------------+--------------+----------------+

    The customization allows
    - to change the foreground or background color;
    - to set or unset the attributes 'bold', 'intense', 'underline';
    - to clear all attributes.

    Customization is done passing a color_spec argument.
    This flag may be provided multiple times.

    The syntax is the following:

    color_spec = face-name + ':' + attributes
    attributes = attribute
               | attribute + ':' + attributes
    attribute  = ('foreground' | 'background') + ':' + color
               | (<empty> | 'no') + font-flag
               | 'none'
    font-flag  = 'italic'
               | 'bold'
               | 'intense'
               | 'underline'
    color      = 'none'
               | [0-255]
               | [0-255] + ',' + [0-255] + ',' + [0-255]
               | ('black', 'blue', 'green', 'red',
                  'cyan', 'magenta', 'yellow', 'white')

    For example, the color_spec

        'refine-added:background:blue:bold'

    sets the color of unique added segments with
    a blue background, written with a bold font.

**\-\-line-numbers** \<compact|aligned\>
    Display line numbers. Style is optional.
    When style = 'compact', take as little width as possible.
    When style = 'aligned', align to tab stops (useful if tab is used for indentation). [default: compact]

**-h**, **\-\-help**
        Prints help information

**-V**, **\-\-version**
        Prints version information

# AUTHOR
Nathan Moreau \<nathan.moreau@m4x.org\>

# LICENSE
The MIT License (MIT)
