"""
Attack Simulator for Hydra-IDS - Production Grade

Simulates network traffic by streaming data from the test dataset with:
- Dynamic label mapping from artifacts
- Real-time attack injection
- Metrics tracking
- Memory-efficient chunked loading
"""

import pandas as pd
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Generator, Optional, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttackSimulator:
    def __init__(
        self,
        data_path: Path,
        labels_path: Path,
        label_mapping_path: Optional[Path] = None,
        chunk_size: int = None
    ):
        """
        Initialize the Attack Simulator.
        
        Args:
            data_path: Path to X_test.csv (features)
            labels_path: Path to y_test.csv (labels)
            label_mapping_path: Path to label_names.json (optional, loads from artifacts)
            chunk_size: If specified, load data in chunks for memory efficiency
        """
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.chunk_size = chunk_size
        self.data = None
        self.labels = None
        
        # Load label mapping
        self.attack_mapping = self._load_label_mapping(label_mapping_path)
        
        # Load data
        self._load_data()
    
    def _load_label_mapping(self, label_mapping_path: Optional[Path]) -> Dict[int, str]:
        """
        Load label mapping from preprocessing artifacts.
        
        Args:
            label_mapping_path: Path to label_names.json
            
        Returns:
            Dict mapping label IDs to names
        """
        if label_mapping_path and Path(label_mapping_path).exists():
            logger.info(f"Loading label mapping from {label_mapping_path}")
            with open(label_mapping_path, 'r') as f:
                label_names = json.load(f)
            
            # Convert to {id: name} dictionary
            mapping = {i: name for i, name in enumerate(label_names)}
            logger.info(f"Loaded {len(mapping)} label classes")
            return mapping
        else:
            # Fallback to default CICIDS mapping
            logger.warning("Label mapping file not found, using default CICIDS mapping")
            return {
                0: "Benign",
                1: "Bot",
                2: "DDoS",
                3: "DoS GoldenEye",
                4: "DoS Hulk",
                5: "DoS Slowhttptest",
                6: "DoS slowloris",
                7: "FTP-Patator",
                8: "Heartbleed",
                9: "Infiltration",
                10: "PortScan",
                11: "SSH-Patator",
                12: "Web Attack - Brute Force",
                13: "Web Attack - Sql Injection",
                14: "Web Attack - XSS"
            }
    
    def _load_data(self):
        """Load test data into memory or prepare for chunked loading."""
        logger.info(f"Loading test data from {self.data_path}...")
        try:
            if self.chunk_size:
                # Don't load all at once, will use chunk loading
                logger.info(f"Prepared for chunked loading with chunk_size={self.chunk_size}")
                self.data = None
                self.labels = None
            else:
                # Load everything
                self.data = pd.read_csv(self.data_path)
                self.labels = pd.read_csv(self.labels_path).squeeze()
                logger.info(f"Loaded {len(self.data)} samples")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def get_attack_types(self) -> Dict[str, int]:
        """Return available attack types and their counts."""
        if self.labels is None:
            # Need to load labels first
            self.labels = pd.read_csv(self.labels_path).squeeze()
        
        counts = self.labels.value_counts().to_dict()
        return {self.attack_mapping.get(k, f"Unknown-{k}"): v for k, v in counts.items()}
    
    def stream_traffic(
        self,
        attack_type: Optional[str] = None,
        count: int = 10,
        interval: float = 0.1,
        random_seed: int = 42,
        apply_evasion: bool = False,
        evasion_epsilon: float = 0.05
    ) -> Generator[Dict, None, None]:
        """
        Stream traffic from the dataset.
        
        Args:
            attack_type: Specific attack name to simulate (e.g., "DDoS"). If None, mix.
            count: Number of packets to stream.
            interval: Delay between packets in seconds.
            random_seed: Random seed for reproducibility.
            apply_evasion: If True, apply evasion noise in real-time
            evasion_epsilon: Epsilon for evasion perturbations
            
        Yields:
            dict: {
                'features': DataFrame (1 row),
                'true_label_id': int,
                'true_label_name': str,
                'timestamp': float,
                'evasion_applied': bool
            }
        """
        np.random.seed(random_seed)
        rng = np.random.default_rng(random_seed)
        
        # Load data if not already loaded
        if self.data is None:
            self.data = pd.read_csv(self.data_path)
            self.labels = pd.read_csv(self.labels_path).squeeze()
        
        # Filter indices based on attack_type
        if attack_type:
            # Find ID associated with name
            target_id = None
            for aid, name in self.attack_mapping.items():
                if name.lower() == attack_type.lower():
                    target_id = aid
                    break
            
            if target_id is None:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            indices = self.labels[self.labels == target_id].index
            if len(indices) == 0:
                logger.warning(f"No samples found for attack type: {attack_type}")
                return
        else:
            indices = self.labels.index
        
        # Sample indices
        if len(indices) > 0:
            selected_indices = np.random.choice(indices, size=min(count, len(indices)), replace=False)
        else:
            return
        
        logger.info(f"Starting simulation: {count} packets of type '{attack_type or 'Mixed'}'")
        
        for idx in selected_indices:
            row = self.data.iloc[[pd.Index(self.data.index).get_loc(idx)]]  # Keep DataFrame format
            label_id = self.labels.iloc[pd.Index(self.data.index).get_loc(idx)]
            label_name = self.attack_mapping.get(label_id, "Unknown")
            
            # Apply evasion if requested
            evasion_applied = False
            if apply_evasion and label_id != 0:  # Don't evade benign traffic
                signs = rng.choice([-1, 1], size=row.shape[1])
                perturbation = signs * evasion_epsilon
                row = row + perturbation
                evasion_applied = True
            
            packet = {
                'features': row,
                'true_label_id': int(label_id),
                'true_label_name': label_name,
                'timestamp': time.time(),
                'evasion_applied': evasion_applied
            }
            
            yield packet
            time.sleep(interval)
    
    def create_attack_scenario(
        self,
        scenario_name: str = "mixed_campaign"
    ) -> Generator[Dict, None, None]:
        """
        Stream a predefined attack scenario.
        
        Scenarios:
        - 'mixed_campaign': Mix of different attacks
        - 'dos_campaign': Primarily DoS attacks
        - 'web_campaign': Web attacks
        - 'brute_force': Brute force attacks
        
        Args:
            scenario_name: Name of scenario to simulate
            
        Yields:
            Traffic packets
        """
        logger.info(f"Starting attack scenario: {scenario_name}")
        
        scenarios = {
            'mixed_campaign': [
                ('DDoS', 20),
                ('PortScan', 15),
                ('Bot', 10),
                ('Benign', 5)
            ],
            'dos_campaign': [
                ('DoS Hulk', 25),
                ('DDoS', 15),
                ('DoS slowloris', 10)
            ],
            'web_campaign': [
                ('Web Attack - Brute Force', 15),
                ('Web Attack - XSS', 10),
                ('Web Attack - Sql Injection', 10)
            ],
            'brute_force': [
                ('FTP-Patator', 20),
                ('SSH-Patator', 20)
            ]
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")
        
        for attack_type, count in scenarios[scenario_name]:
            for packet in self.stream_traffic(attack_type=attack_type, count=count, interval=0.05):
                yield packet


if __name__ == "__main__":
    # Test
    DATA_DIR = Path(".") / "data/processed"
    LABEL_MAPPING = Path(".") / "models/preprocessing/label_names.json"
    
    if (DATA_DIR / "X_test.csv").exists():
        sim = AttackSimulator(
            data_path=DATA_DIR / "X_test.csv",
            labels_path=DATA_DIR / "y_test.csv",
            label_mapping_path=LABEL_MAPPING if LABEL_MAPPING.exists() else None
        )
        print("Available attacks:", sim.get_attack_types())
        
        print("\nStreaming 5 mixed packets:")
        for packet in sim.stream_traffic(count=5, interval=0.01):
            print(f"- Sent: {packet['true_label_name']}, Evaded: {packet['evasion_applied']}")
    else:
        print("Test data not found")
