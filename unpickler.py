import pickle
import sys
from basic_agent import BaseAgent


class AgentUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            __import__(module)
            mod = sys.modules[module]
            return getattr(mod, name)
        except (ImportError, AttributeError, KeyError):
            if name in ['StrongTicTacToeAgent', 'VeryStrongQAgent', 'MinimaxAgent', 'QAgent']:
                print(f"Creating dynamic replacement for {name}")
                DynamicAgent = type(name, (BaseAgent,), {
                    'get_action': lambda self, board: self._default_action(board),
                    '_default_action': lambda self, board: next(
                        ((r, c) for r in range(3) for c in range(3) if board[r, c] == 0),
                        (0, 0)
                    ),
                    'reset': lambda self: None,
                    'update': lambda self, *args, **kwargs: None
                })
                return DynamicAgent
            print(f"Creating generic placeholder for {name}")
            return type(name, (), {})


def load_agent(filepath, player_id=None):
    """Load agent with robust error handling"""
    try:
        print(f"Loading agent from {filepath}...")
        with open(filepath, 'rb') as f:
            unpickler = AgentUnpickler(f)
            agent = unpickler.load()

            if player_id is not None:
                print(f"Setting player_id to {player_id}")
                agent.player_id = player_id

            if not hasattr(agent, 'get_action') or not callable(agent.get_action):
                print(f"Warning: Agent doesn't have a proper get_action method.")
                agent.get_action = lambda board: next(
                    ((r, c) for r in range(3) for c in range(3) if board[r, c] == 0),
                    (0, 0)
                )

            if not hasattr(agent, 'reset'):
                agent.reset = lambda: None

            if not hasattr(agent, 'update'):
                agent.update = lambda *args, **kwargs: None

            return agent
    except Exception as e:
        print(f"Error loading agent from {filepath}: {e}")
        print("Using fallback agent instead.")

        fallback = BaseAgent(player_id=player_id if player_id is not None else 1)
        return fallback
