from tabnanny import verbose
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, FileReadTool
from pydantic import BaseModel, Field
from typing import List
#from .tools.push_tool import PushNotificationTool
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

class Investment(BaseModel):
    type: str = Field(description="Type of investment. Could be Stock, Property or Crypto")
    code: str = Field(description="A short code.eg: Ticker for a stock or ETF. Keep blank for property")
    current_value: str = Field(description="Current value in AUD if existing investment. 0 otherwise")
    rating: str = Field(description="Assessed rating out of 5, Anything less than 3 is bad")
    reason: str = Field(description="Reason for the rating")
    action: str = Field(description="Buy, Sell or Hold. Use only these 3 options and nothing else")

class Portfolio(BaseModel):
    investments: List[Investment] = Field(description="List of investments")

class TrendingCompany(BaseModel):
    """ A Company that is in the news and attracting attention"""
    name: str = Field(description="Company Name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")

class TrendingCompanyList(BaseModel):
    """ List of ,ultiple trending companies that are in the news"""
    companies: List[TrendingCompany] = Field(description="List of companies trending in the news")

class TrendingCompanyResearch(BaseModel):
    """ Detailed research on a company"""
    name: str = Field(description="Company name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth opportunities")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """ A list of detailed research on all the companies """
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")

@CrewBase
class Advisor():
    """Advisor crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def portfolio_analyser(self) -> Agent:
        return Agent(config=self.agents_config['portfolio_analyser'],
                     tools=[SerperDevTool(),FileReadTool(file_path='/Users/km/AI/projects/claude-proj/financial_advisor/advisor/src/advisor/Inputs/Investment-Portfolio.csv')], 
                     memory=True,
                     verbose=True)
    
    @task
    def portfolio_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_analysis'],
            output_pydantic=Portfolio,
            verbose=True
        )

    @agent
    def portfolio_curator(self) -> Agent:
        return Agent(config=self.agents_config['portfolio_curator'],
                     memory=True,
                     verbose=True)
    
   
    
    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(config=self.agents_config['trending_company_finder'],
                    tools=[SerperDevTool()], memory=True)
       
    @agent
    def financial_researcher(self) -> Agent:
        return Agent(config=self.agents_config['financial_researcher'], 
                    tools=[SerperDevTool()])

    @agent
    def stock_picker(self) -> Agent:
        return Agent(config=self.agents_config['stock_picker'], 
                        memory=True)
    
    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList,
            
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList,
           # context=[self.find_trending_companies] # <-- Chain here
        )

    @task
    def pick_best_company(self) -> Task:
        print(self.tasks_config)
        return Task(
            config=self.tasks_config['pick_best_company'],
            #context=[self.research_trending_companies] # <-- Chain here instead of 'input='
        )
        
    @task
    def generate_final_rebalance_report(self) -> Task:
        return Task(
            config=self.tasks_config['generate_final_rebalance_report'],
            verbose=True,
            #context=[self.portfolio_analysis, self.pick_best_company], # <-- Feeds both paths into the final report
        )


    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker crew"""
        

        manager = Agent(
            config=self.agents_config['manager'],
            memory=True,  
            allow_delegation=True,
          )
       
        
            
        return Crew(
                agents=self.agents,
                tasks=self.tasks, 
                process=Process.hierarchical,
                verbose=True,
                manager_agent=manager,
                memory=True,
                # Long-term memory for persistent storage across sessions
                long_term_memory = LongTermMemory(
                    storage=LTMSQLiteStorage(
                        db_path="./memory/long_term_memory_storage.db"
                    )
                ),
                # Short-term memory for current context using RAG
                short_term_memory = ShortTermMemory(
                    storage = RAGStorage(
                            embedder_config={
                                "provider": "openai",
                                "config": {
                                    "model_name": 'text-embedding-3-small',  # <-- Change "model" to "model_name
                                    "api_key_env_var": "OPENAI_API_KEY"  # <-- Tell Chroma to look for your standard key
                                }
                            },
                            type="short_term",
                            path="./memory/"
                        )
                    ),            # Entity memory for tracking key information about entities
                entity_memory = EntityMemory(
                    storage=RAGStorage(
                        embedder_config={
                            "provider": "openai",
                            "config": {
                               "model_name": 'text-embedding-3-small',  # <-- Change "model" to "model_name
                               "api_key_env_var": "OPENAI_API_KEY"
                            }
                        },
                        type="short_term",
                        path="./memory/"
                    )
                ),
            )
