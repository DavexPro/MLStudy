#!/usr/bin/env python3
# coding=utf-8
# -*- utf8 -*-
# author=dave.fang@outlook.com
# create=20170427

import logging
import sys

LOGGER = logging.getLogger("gaLog")

LOGGER_HANDLER = logging.StreamHandler(sys.stdout)

FORMATTER = logging.Formatter("\r[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

LOGGER_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(LOGGER_HANDLER)
LOGGER.setLevel(logging.INFO)