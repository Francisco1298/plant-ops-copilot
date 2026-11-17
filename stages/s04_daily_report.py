from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model
from app.tools import query_kpis

DATE = "2026-10-19"

# 1. Get the data (no AI here: just the tool reading the warehouse)
data = query_kpis.invoke({
    "start_date": DATE,
    "end_date": DATE,
    "metrics": ["estado_operacion", "biogas_nm3", "ch4_pct_prom", "electricidad_potencial_kwh", "temp_min_c"],
})
print("DATA FROM THE WAREHOUSE:\n", data, "\n")

# 2. The prompt template: fixed instructions + gaps to fill ({date}, {data})
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You write the daily report for the manager of a biogas plant.\n"
     "Rules:\n"
     "1. Write in English, maximum 5 lines.\n"
     "2. First line: the day's status (NORMAL, ALERTA or CRITICO).\n"
     "3. Include biogas (Nm3), CH4 (%) and POTENTIAL electricity (kWh). It is potential, not generated.\n"
     "4. Use ONLY the numbers in the data. Never invent figures.\n"
     "5. If the status is not NORMAL, add one line with the recommended action.\n"
     ),

    ("human", "Plant data for {date}:\n{data}\n\nWrite the daily report."),
])

# 3. The chain: prompt -> model -> clean text
chain = prompt | get_chat_model() | StrOutputParser()
print("REPORT:\n", chain.invoke({"date": DATE, "data": data}))