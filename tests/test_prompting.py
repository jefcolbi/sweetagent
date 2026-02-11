from unittest import TestCase
from pathlib import Path
import os

from sweetagent.core import WorkMode
from sweetagent.io.base import ConsoleStaIO
from sweetagent.llm_agent import LLMAgent
from sweetagent.llm_client import LLMClient
from decouple import config
from datetime import date

from sweetagent.short_term_memory.session import SessionMemory

cur_dir = Path(os.path.abspath(__file__)).parent
src_path = cur_dir.parent / "src"
print(f"{src_path = }")

from sweetagent.prompt import PromptEngine, BaseState, FSMPromptEngine, Transition

LLM_PROVIDER = config("LLM_PROVIDER", default="azure")
LLM_MODEL = config("LLM_MODEL", default="gpt-4o")
LLM_API_KEYS = config("AZURE_API_KEYS").split(",")


class PromptEngineTestCase(TestCase):
    def test_01_system_message(self):
        engine = PromptEngine()
        engine.native_tool_call_support = False
        engine.native_thought = True
        engine.agent_name = "Support Agent"
        engine.agent_role = "Provide support to customers"
        engine.user_full_name = "Jeffersson Mattersson"
        engine.user_extra_infos = {"age": 10}
        engine.agent_steps = [
            "Ask the user where he is going",
            "Find the price for this destination",
            "Book the travel",
        ]
        print(engine.get_system_message(with_tools=[{"yes": "no"}]))

    def test_02_decode_simple_message(self):
        example = """+++ thought +++
The user is requesting a Python program to compute the Fibonacci sequence. I will provide a simple implementation using an iterative approach.
+++ kind +++
message
+++ message +++
Certainly! Below is a Python program to compute the Fibonacci sequence up to a specified number of terms:

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_sequence = [0, 1]
    for i in range(2, n):
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)

    return fib_sequence

# Example usage
num_terms = 10
result = fibonacci(num_terms)
print(f"Fibonacci sequence with {num_terms} terms: {result}")

This program defines a function fibonacci that takes an integer n as input and returns a list containing the first n numbers of the Fibonacci sequence.
+++ data +++

+++ tool_name +++

+++ tool_arguments +++

+++ end +++"""

        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)


class _IdleState(BaseState):
    name = "Idle"


class _WorkingState(BaseState):
    name = "Working"
    default_entry = "on_enter"
    default_do = "run"
    default_exit = "cleanup"


class BaseStateTestCase(TestCase):
    def test_01_minimal_state_declaration(self):
        state = _IdleState()
        self.assertEqual(state.to_string(), "state Idle")

    def test_02_state_declaration_with_actions(self):
        state = _WorkingState()
        expected = (
            "state Working {\n  entry / on_enter()\n  do / run()\n  exit / cleanup\n}"
        )
        self.assertEqual(state.to_string(), expected)

    def test_03_transition_includes_condition_and_action(self):
        state = _IdleState()
        state.add_transition(
            event="start",
            condition="ready",
            action="log",
            next_state="Running",
        )
        expected = "state Idle\n\nIdle --> Running : start [ready] / log\n"
        self.assertEqual(state.to_string(), expected)

    def test_03_decode_message_with_data(self):
        example = """+++ thought +++
Maybe the user wants to send an email. I am will ask him
+++ kind +++
message
+++ message +++
Do you want me so send an email ?
+++ data +++
choices:
  - Yes
  - No
+++ end +++"""
        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)

    def test_04_decode_tool_call(self):
        example = """+++ thought +++
The user wants to know the current weather in Douala. I will use the get_weather tool to retrieve this information.
+++ kind +++
tool_call
+++ tool_name +++
get_weather
+++ tool_arguments +++
~~~~ city ~~~~
Douala

+++ end +++"""

        engine = PromptEngine()
        engine.extract_formatted_llm_response(example)


class _AskNameState(BaseState):
    name = "AskName"
    event_name_provided = "name_provided"
    default_transitions = [
        Transition(event=event_name_provided, next_state="AskAge"),
    ]


class _AskAgeState(BaseState):
    name = "AskAge"
    event_age_provided = "age_provided"
    default_transitions = [
        Transition(event=event_age_provided, next_state="ShowSummary"),
    ]


class _ShowSummaryState(BaseState):
    name = "ShowSummary"
    event_done = "done"
    default_transitions = [
        Transition(event=event_done, next_state="End"),
    ]


class _EndState(BaseState):
    name = "End"


class _AccueilState(BaseState):
    name = "Accueil"
    default_entry = "saluer_en_se_presentant_poliment_et_professionellement_et_ensuite_demander_prenom"
    event_message_utilisateur = "message_utilisateur"
    default_transitions = [
        Transition(event=event_message_utilisateur, next_state="DemandePrenom"),
    ]


class _DemandePrenomState(BaseState):
    name = "DemandePrenom"
    default_entry = "demander_prenom"
    event_reponse_invalide = "reponse_invalide"
    event_prenom_valide = "prenom_valide"
    default_transitions = [
        Transition(event=event_reponse_invalide, next_state="DemandePrenom"),
        Transition(event=event_prenom_valide, next_state="DemandeService"),
    ]


class _DemandeServiceState(BaseState):
    name = "DemandeService"
    default_entry = "demander_service_avec_options:  Réparations rapides :: Rénovation énergétique :: Travaux électriques :: Salle de bain :: Cuisine"
    event_service_invalide = "service_invalide"
    event_service_valide = "service_valide"
    default_transitions = [
        Transition(event=event_service_invalide, next_state="DemandeService"),
        Transition(event=event_service_valide, next_state="DemandeDelai"),
    ]


class _DemandeDelaiState(BaseState):
    name = "DemandeDelai"
    default_entry = "demander_delai_execution"
    event_delai_invalide = "delai_invalide"
    event_delai_valide = "delai_valide"
    default_transitions = [
        Transition(event=event_delai_invalide, next_state="DemandeDelai"),
        Transition(event=event_delai_valide, next_state="CalculDate"),
    ]


class _CalculDateState(BaseState):
    name = "CalculDate"
    default_entry = "appeler_outil_getCurrentDay_et_calculer_date"
    default_transitions = [
        Transition(next_state="DemandeBudget"),
    ]


class _DemandeBudgetState(BaseState):
    name = "DemandeBudget"
    default_entry = "demander_budget_en_euros"
    event_budget_invalide = "budget_invalide"
    event_budget_valide = "budget_valide"
    default_transitions = [
        Transition(event=event_budget_invalide, next_state="DemandeBudget"),
        Transition(event=event_budget_valide, next_state="DemandeAdresse"),
    ]


class _DemandeAdresseState(BaseState):
    name = "DemandeAdresse"
    default_entry = "demander_adresse_charleroi"
    event_adresse_invalide = "adresse_invalide"
    event_adresse_valide = "adresse_valide"
    default_transitions = [
        Transition(event=event_adresse_invalide, next_state="DemandeAdresse"),
        Transition(event=event_adresse_valide, next_state="Recapitulatif"),
    ]


class _RecapitulatifState(BaseState):
    name = "Recapitulatif"
    default_entry = "afficher_resume_et_demander_confirmation"
    event_confirmation_oui = "confirmation_oui"
    event_confirmation_non = "confirmation_non"
    default_transitions = [
        Transition(event=event_confirmation_oui, next_state="Sauvegarde"),
        Transition(event=event_confirmation_non, next_state="Correction"),
    ]


class _CorrectionState(BaseState):
    name = "Correction"
    default_entry = "demander_information_erronee"
    event_correction_prenom = "correction_prenom"
    event_correction_service = "correction_service"
    event_correction_delai = "correction_delai"
    event_correction_budget = "correction_budget"
    event_correction_adresse = "correction_adresse"
    default_transitions = [
        Transition(event=event_correction_prenom, next_state="DemandePrenom"),
        Transition(event=event_correction_service, next_state="DemandeService"),
        Transition(event=event_correction_delai, next_state="DemandeDelai"),
        Transition(event=event_correction_budget, next_state="DemandeBudget"),
        Transition(event=event_correction_adresse, next_state="DemandeAdresse"),
    ]


class _SauvegardeState(BaseState):
    name = "Sauvegarde"
    default_entry = "sauvegarder_donnees_avec_outil_sauvegarder_et_feliciter"


class SupportAgent(LLMAgent):
    def getCurrentDay(self):
        return str(date.today())

    def sauvegarder(
        self,
        first_name: str = None,
        service: str = None,
        target_date: str = None,
        budget: float = None,
        address: str = None,
    ):
        print(f"sauvegarde de ...")
        return "Informations sauvegardées avec succès"


class FSMPromptEngineTestCase(TestCase):
    def test_01_system_message_contains_fsm(self):
        ask_name = _AskNameState()
        ask_age = _AskAgeState()
        show_summary = _ShowSummaryState()
        end = _EndState()

        engine = FSMPromptEngine(
            initial_state=ask_name,
            end_state=end,
            states=[ask_name, ask_age, show_summary, end],
        )

        message = engine.get_system_message()
        self.assertIsInstance(message, str)
        self.assertIn("@startuml", message)
        self.assertIn("[*] --> AskName", message)
        self.assertIn("AskName --> AskAge : name_provided", message)
        self.assertIn("AskAge --> ShowSummary : age_provided", message)
        self.assertIn("ShowSummary --> End : done", message)
        self.assertIn("End --> [*]", message)

        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER,
            LLM_MODEL,
            config("OPENAI_API_KEYS").split(","),
            stdio=stdio,
            completion_kwargs={"temperature": 1},
        )
        agent = SupportAgent(
            "Support Agent",
            "Collect user profile and summarize it",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            prompt_engine=engine,
            work_mode=WorkMode.CHAT,
        )
        agent.run("Hi")

    def test_02_system_message_contains_full_flow(self):
        accueil = _AccueilState()
        demande_prenom = _DemandePrenomState()
        demande_service = _DemandeServiceState()
        demande_delai = _DemandeDelaiState()
        calcul_date = _CalculDateState()
        demande_budget = _DemandeBudgetState()
        demande_adresse = _DemandeAdresseState()
        recapitulatif = _RecapitulatifState()
        correction = _CorrectionState()
        sauvegarde = _SauvegardeState()

        engine = FSMPromptEngine(
            initial_state=accueil,
            end_state=sauvegarde,
            states=[
                accueil,
                demande_prenom,
                demande_service,
                demande_delai,
                calcul_date,
                demande_budget,
                demande_adresse,
                recapitulatif,
                correction,
                sauvegarde,
            ],
        )

        message = engine.get_system_message()
        # self.assertIn("[*] --> Accueil", message)
        # self.assertIn("state Accueil", message)
        # self.assertIn("entry / saluer_et_demander_prenom()", message)
        # self.assertIn("Accueil --> DemandePrenom : message_utilisateur", message)
        #
        # self.assertIn("state DemandePrenom", message)
        # self.assertIn("entry / demander_prenom()", message)
        # self.assertIn("DemandePrenom --> DemandePrenom : reponse_invalide", message)
        # self.assertIn("DemandePrenom --> DemandeService : prenom_valide", message)
        #
        # self.assertIn("state DemandeService", message)
        # self.assertIn("entry / demander_service()", message)
        # self.assertIn("DemandeService --> DemandeService : service_invalide", message)
        # self.assertIn("DemandeService --> DemandeDelai : service_valide", message)
        #
        # self.assertIn("state DemandeDelai", message)
        # self.assertIn("entry / demander_delai_execution()", message)
        # self.assertIn("DemandeDelai --> DemandeDelai : delai_invalide", message)
        # self.assertIn("DemandeDelai --> CalculDate : delai_valide", message)
        #
        # self.assertIn("state CalculDate", message)
        # self.assertIn("entry / appeler_outil_currentDay_et_calculer_date()", message)
        # self.assertIn("CalculDate --> DemandeBudget", message)
        #
        # self.assertIn("state DemandeBudget", message)
        # self.assertIn("entry / demander_budget()", message)
        # self.assertIn("DemandeBudget --> DemandeBudget : budget_invalide", message)
        # self.assertIn("DemandeBudget --> DemandeAdresse : budget_valide", message)
        #
        # self.assertIn("state DemandeAdresse", message)
        # self.assertIn("entry / demander_adresse_charleroi()", message)
        # self.assertIn("DemandeAdresse --> DemandeAdresse : adresse_invalide", message)
        # self.assertIn("DemandeAdresse --> Recapitulatif : adresse_valide", message)
        #
        # self.assertIn("state Recapitulatif", message)
        # self.assertIn(
        #     "entry / afficher_resume_et_demander_confirmation()", message
        # )
        # self.assertIn("Recapitulatif --> Sauvegarde : confirmation_oui", message)
        # self.assertIn("Recapitulatif --> Correction : confirmation_non", message)
        #
        # self.assertIn("state Correction", message)
        # self.assertIn("entry / demander_information_erronee()", message)
        # self.assertIn("Correction --> DemandePrenom : correction_prenom", message)
        # self.assertIn("Correction --> DemandeService : correction_service", message)
        # self.assertIn("Correction --> DemandeDelai : correction_delai", message)
        # self.assertIn("Correction --> DemandeBudget : correction_budget", message)
        # self.assertIn("Correction --> DemandeAdresse : correction_adresse", message)
        #
        # self.assertIn("state Sauvegarde", message)
        # self.assertIn("entry / sauvegarder_donnees_et_feliciter()", message)
        # self.assertIn("Sauvegarde --> [*]", message)

        stdio = ConsoleStaIO("default")
        client = LLMClient(
            LLM_PROVIDER,
            LLM_MODEL,
            config("OPENAI_API_KEYS").split(","),
            stdio=stdio,
            completion_kwargs={"temperature": 1},
        )
        agent = SupportAgent(
            "Fix & Renove Charleroi Agent",
            "Ton role est de collecter les informations sur l'utilisateur et son besoin. L'utilisateur parle exclusivement français donc tu vas converser avec lui uniquement en français.",
            client,
            short_term_memory=SessionMemory(),
            stdio=stdio,
            prompt_engine=engine,
            work_mode=WorkMode.CHAT,
        )
        agent.register_function_as_tool(agent.getCurrentDay)
        agent.register_function_as_tool(agent.sauvegarder)
        agent.run("salut")
