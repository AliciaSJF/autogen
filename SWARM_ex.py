from typing import Any, Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

# 🔹 Herramienta para procesar reembolsos
def refund_flight(flight_id: str) -> str:
    """Simula el reembolso de un vuelo"""
    return f"Flight {flight_id} refunded"

# 🔹 Configuración del cliente de modelo
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# 🔹 Definición de agentes
travel_agent = AssistantAgent(
    "travel_agent",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="""
    You are a travel agent.
    The flights_refunder is in charge of refunding flights.
    If you need information from the user, first send your message, then handoff to the user.
    Use TERMINATE when the travel planning is complete.
    """,
)

flights_refunder = AssistantAgent(
    "flights_refunder",
    model_client=model_client,
    handoffs=["travel_agent", "user"],
    tools=[refund_flight],
    system_message="""
    You are an agent specialized in refunding flights.
    You only need flight reference numbers to refund a flight.
    Use the refund_flight tool to process refunds.
    If you need information from the user, first send your message, then handoff to the user.
    When the transaction is complete, handoff to the travel agent to finalize.
    """,
)

# 🔹 Condiciones de terminación
termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")

# 🔹 Creación del equipo Swarm
team = Swarm([travel_agent, flights_refunder], termination_condition=termination)

# 🔹 Tarea inicial
task = "I need to refund my flight."

async def run_team_stream():
    task_result = await Console(team.run_stream(task=task))
    last_message = task_result.messages[-1]

    # 🔹 Si el agente transfiere al usuario, espera entrada manual
    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")
        task_result = await Console(
            team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

# 🔹 Ejecutar el sistema
asyncio.run(
     run_team_stream()
)