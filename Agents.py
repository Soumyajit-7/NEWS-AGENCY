import json
import os

import requests
from langchain.tools import tool

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from Utils import print_agent_output
from Utils import GROQ_LLM
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

#search_tool = DuckDuckGoSearchRun()
search_tool = SerperDevTool()
class NewsAgents():
    def make_news_scraper_agent(self):
        return Agent(
            role='News Scraper Agent',
            goal="""Collect news data from various sources like news websites, social media, and official press releases. 
            Gather the latest news articles, tweets, and other relevant information.""",
            backstory="""You are an expert at collecting up-to-date and relevant news from a variety of sources efficiently and effectively.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            tools=[search_tool],
            step_callback=lambda x: print_agent_output(x, "News Scraper Agent"),
        )
    
    def search_news_scraper_agent(self):
        return Agent(
            role='News Scraper Agent',
            goal="""Take in a news from the user. \
                Collect all the data on that particular news from various sources like news websites, social media, and official press releases . \
                Try to keep the sources trusted and reliablefrom where the news is being scraped.""",
            backstory="""You are an expert at collecting up-to-date and relevant data and information about a particular news from a \
                variety of sources efficiently and effectively.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            tools=[search_tool],
            step_callback=lambda x: print_agent_output(x, "S_News Scraper Agent"),
        )

    def make_event_detection_agent(self):
        return Agent(
            role='Event Detection Agent',
            goal="""Identify significant news events from the collected data. Use NLP to process and analyze the gathered data, 
            clustering similar news items to identify significant events.""",
            backstory="""You are a master at recognizing and clustering significant news events from vast amounts of data using advanced NLP techniques.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Event Detection Agent"),
        )

    def make_fact_checking_agent(self):
        return Agent(
            role='Fact-Checking Agent',
            goal="""Verify the accuracy of the news by cross-referencing with reliable sources. 
            Search reputable databases, official statements, and trusted news sources to confirm the facts in the detected news events.""",
            backstory="""You excel at cross-referencing information and confirming the accuracy of news events by using reliable and trusted sources.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            tools=[search_tool],
            step_callback=lambda x: print_agent_output(x, "Fact-Checking Agent"),
        )

    def Search_make_fact_checking_agent(self):
        return Agent(
            role='Fact-Checking Agent',
            goal="""Verify the accuracy of the news by cross-referencing with reliable sources. \
                Search reputable databases, official statements, and trusted news sources to confirm the facts in the detected news events.""",
            backstory="""You excel at cross-referencing information and confirming the accuracy of news events by using reliable and \
                trusted sources.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            tools=[search_tool],
            step_callback=lambda x: print_agent_output(x, "S_Fact-Checking Agent"),
        )

    def make_content_generation_agent(self):
        return Agent(
            role='Content Generation Agent',
            goal="""Write news articles based on verified information. 
            Use LLMs to generate coherent and engaging news articles, ensuring the tone and style are appropriate for the target audience.""",
            backstory="""You are skilled at crafting well-written, engaging news articles that are based on verified facts and tailored to the target audience.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Content Generation Agent"),
        )

    def make_editor_agent(self):
        return Agent(
            role='Editor Agent',
            goal="""Review and edit the generated articles for quality and consistency. 
            Check for grammar, clarity, coherence, and adherence to editorial guidelines.""",
            backstory="""You have a keen eye for detail and ensure that all news articles are of the highest quality, consistent, and adhere to editorial standards.""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x, "Editor Agent"),
        )

