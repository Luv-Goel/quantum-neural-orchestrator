"""
Orchestration Layer for Autonomous Systems

This module provides:
- Multi-agent coordination
- Task scheduling and distribution
- System health monitoring
- Decision-making framework
"""

import json
import time
import threading
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

class AgentStatus(Enum):
    """Status enum for agents"""
    IDLE = auto()
    WORKING = auto()
    BLOCKED = auto()
    FAILED = auto()
    COMPLETED = auto()

@dataclass
class Agent:
    """Represents an autonomous agent in the system"""
    agent_id: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    task_queue: queue.Queue = field(default_factory=queue.Queue)
    last_activity: float = time.time()
    performance_score: float = 1.0

    def update_status(self, new_status: AgentStatus):
        """Update the agent's status"""
        self.status = new_status
        self.last_activity = time.time()

    def assign_task(self, task: str):
        """Assign a new task to the agent"""
        self.task_queue.put(task)
        if self.status == AgentStatus.IDLE:
            self.update_status(AgentStatus.WORKING)
            self.current_task = self.task_queue.get()

    def complete_task(self):
        """Mark current task as completed"""
        self.current_task = None
        if self.task_queue.empty():
            self.update_status(AgentStatus.IDLE)
        else:
            self.current_task = self.task_queue.get()

class Orchestrator:
    """Main orchestration class for managing agents and tasks"""
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.task_queue = queue.PriorityQueue()
        self.system_health = 1.0
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def add_agent(self, agent_id: str, capabilities: List[str]):
        """Add a new agent to the system"""
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists")
        self.agents[agent_id] = Agent(agent_id, capabilities)

    def remove_agent(self, agent_id: str):
        """Remove an agent from the system"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        del self.agents[agent_id]

    def submit_task(self, task: str, priority: int = 0):
        """Submit a new task to the system"""
        self.task_queue.put((priority, task))
        self._distribute_tasks()

    def _distribute_tasks(self):
        """Distribute tasks to available agents"""
        while not self.task_queue.empty():
            priority, task = self.task_queue.get()
            
            # Find best agent for the task
            best_agent = None
            best_score = -1
            
            for agent in self.agents.values():
                if agent.status == AgentStatus.IDLE:
                    # Simple scoring based on performance and capabilities
                    score = agent.performance_score
                    if score > best_score:
                        best_score = score
                        best_agent = agent
            
            if best_agent:
                best_agent.assign_task(task)
            else:
                # No available agents, put task back in queue
                self.task_queue.put((priority, task))
                break

    def _monitor_system(self):
        """Monitor system health and agent status"""
        while True:
            time.sleep(5)  # Check every 5 seconds
            
            # Update system health based on agent status
            healthy_agents = 0
            for agent in self.agents.values():
                if agent.status in [AgentStatus.IDLE, AgentStatus.WORKING]:
                    healthy_agents += 1
            
            if len(self.agents) > 0:
                self.system_health = healthy_agents / len(self.agents)
            
            # Check for stalled agents
            current_time = time.time()
            for agent in self.agents.values():
                if (agent.status == AgentStatus.WORKING and
                    current_time - agent.last_activity > 30):  # 30 seconds timeout
                    agent.update_status(AgentStatus.BLOCKED)

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'system_health': self.system_health,
            'agents': {aid: {
                'status': a.status.name,
                'current_task': a.current_task,
                'performance': a.performance_score,
                'queue_size': a.task_queue.qsize()
            } for aid, a in self.agents.items()},
            'pending_tasks': self.task_queue.qsize()
        }

class DecisionEngine:
    """Decision-making engine for the orchestration system"""
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.rules = []

    def add_rule(self, condition: Callable, action: Callable):
        """Add a decision rule"""
        self.rules.append((condition, action))

    def evaluate(self):
        """Evaluate all rules and execute actions"""
        status = self.orchestrator.get_system_status()
        
        for condition, action in self.rules:
            if condition(status):
                action(status)

# Example usage
if __name__ == "__main__":
    # Create orchestration system
    orchestrator = Orchestrator()
    
    # Add agents with different capabilities
    orchestrator.add_agent("agent_ml", ["machine_learning", "data_processing"])
    orchestrator.add_agent("agent_quantum", ["quantum_computing", "feature_mapping"])
    orchestrator.add_agent("agent_io", ["data_ingestion", "result_export"])
    
    # Create decision engine
    decision_engine = DecisionEngine(orchestrator)
    
    # Add a rule: if system health is low, print warning
    decision_engine.add_rule(
        condition=lambda s: s['system_health'] < 0.7,
        action=lambda s: print(f"WARNING: System health low: {s['system_health']}")
    )
    
    # Submit some tasks
    orchestrator.submit_task("process training data", priority=2)
    orchestrator.submit_task("run quantum feature mapping", priority=1)
    orchestrator.submit_task("export results", priority=3)
    
    # Simulate agent work
    def simulate_agent_work(agent_id, duration):
        agent = orchestrator.agents[agent_id]
        agent.update_status(AgentStatus.WORKING)
        print(f"{agent_id} working on {agent.current_task}")
        time.sleep(duration)
        agent.update_status(AgentStatus.COMPLETED)
        print(f"{agent_id} completed {agent.current_task}")
        agent.complete_task()
    
    # Start simulation threads
    threads = []
    for agent_id in orchestrator.agents:
        t = threading.Thread(target=simulate_agent_work, args=(agent_id, 2))
        threads.append(t)
        t.start()
    
    # Wait for threads to complete
    for t in threads:
        t.join()
    
    # Evaluate decisions
    decision_engine.evaluate()
    
    # Print final system status
    print("\nFinal system status:")
    print(json.dumps(orchestrator.get_system_status(), indent=2))