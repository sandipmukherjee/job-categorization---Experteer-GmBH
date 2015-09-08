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


complete_jobs_file = open('complete_jobs.xml')
xml_doc = complete_jobs_file.read()

jobs = ET.fromstring(xml_doc)
jobs_list = list(jobs)

df_columns = ['title', 'desc', 'industry','location','company', 'function', 'career_level']
jobs_df = pd.DataFrame(columns=df_columns)

print len(jobs_list)
i = 0
for job in jobs_list:
    title = job.find('title')
    raw_description = job.find('description')
    soup = BeautifulSoup(raw_description.text)
    description = soup.get_text()
    description = ' '.join(description.split())
    industry = job.find('industry/id')
    function = job.find('function/id')
    career_level = job.find('career_level/id')
    company_id = job.find('company/id')
    location = job.find('locations/location/country')
    #Only take the surrounding text of experience requirement section
    description_mod = ""
    match_find = find_with_context(description, 'experience', 500)
    for match in match_find:
        description_mod += "%s %s %s" % match
    jobs_df.loc[i] = [title.text, description_mod, industry.text,location.text, company_id.text, function.text, career_level.text]
    i = i+1

jobs_df.save('complete_job')
print jobs_df.head()

