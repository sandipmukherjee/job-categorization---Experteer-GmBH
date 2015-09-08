#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
import pandas as pd
import re


def find_with_context(haystack, needle, context_length, escape=True):
    if escape:
        needle = re.escape(needle)
    return re.findall(r'\b(.{,%d})\b(%s)\b(.{,%d})\b' % (context_length, needle, context_length), haystack)


complete_jobs_file = open('incomplete_jobs.xml')
xml_doc = complete_jobs_file.read()

jobs = ET.fromstring(xml_doc)
jobs_list = list(jobs)

df_columns = ['id', 'title', 'desc','location', 'company']
jobs_df = pd.DataFrame(columns=df_columns)

print len(jobs_list)
i = 0
for job in jobs_list:
    id = job.find('id')
    title = job.find('title')
    raw_description = job.find('description')
    soup = BeautifulSoup(raw_description.text)
    description = soup.get_text()
    description = ' '.join(description.split())
    company_id = job.find('company/id')
    location = job.find('locations/location/country')
    description_mod = ""
    match_find = find_with_context(description, 'experience', 500)
    for match in match_find:
        description_mod += "%s %s %s" % match
    # print company_id.text
    # print location.text
    # print description_mod
    jobs_df.loc[i] = [id.text, title.text, description_mod, location.text, company_id.text]
    i = i+1

jobs_df.save('incomplete_job')
print jobs_df.head()

