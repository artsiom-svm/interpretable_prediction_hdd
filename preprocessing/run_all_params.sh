#!/bin/bash

head -n1 ../Data/Original/errorlog_fixed.csv | tr "," "\n" | sed "s/drive_id//" | sed '/^$/d' > errorlog_params.txt
head -n1 ../Data/Original/swaplog.csv | tr "," "\n" | sed "s/drive_id//" | sed '/^$/d' > swaplog_params.txt
head -n1 ../Data/Original/badchip.csv | tr "," "\n" | sed "s/drive_id//" | sed '/^$/d' > badchip_params.txt
