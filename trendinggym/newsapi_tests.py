# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:05:47 2020

@author: Richard
"""

from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='0566dfe86d9c44c6a3bf8ae60eafb8c6')


all_articles = newsapi.get_everything(q='apple',
                                      from_param='2020-04-01',
                                      to='2020-04-29',
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100,
                                      page=1)

authors = []

for art in all_articles["articles"]:
    authors.append(art["source"]["id"])
    
    

authors = list(set(authors))
