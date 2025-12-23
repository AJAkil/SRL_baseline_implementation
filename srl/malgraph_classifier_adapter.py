#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration adapter between SRL-MalGraph RL environment and existing MalGraph classifier.

This module wraps your existing MalgraphServerFeature/DirectMalgraphClient
to work seamlessly with the SRL environment.

Author: Md Ajwad Akil
Date: December 2025
"""

import sys
from pathlib import Path
import os

# Add MalGuise base path
p = Path(os.path.abspath(__file__))
malguise_base = str(p.parents[2] / "GenAI_Malware_Repository/development_code/asm_pe_file_exp/MalGuise/MalGuise")
sys.path.append(malguise_base)

from src.classifier.models.malgraph.MalgraphModel import MalgraphServerFeature, DirectMalgraphClient, MalgraphModelParams
import torch
import omegaconf


class SRLMalGraphClassifierAdapter:
    """
    Adapter to use your existing MalGraph classifier with SRL environment.
    
    This wraps MalgraphServerFeature/DirectMalgraphClient to provide
    a simple predict(acfg_json) interface.
    """
    
    def __init__(
        self,
        use_direct_client: bool = True,
        threshold_type: str = '100fpr',
        device: torch.device = None,
        server_port: int = 5001
    ):
        """
        Initialize adapter.
        
        Args:
            use_direct_client: Use DirectMalgraphClient (True) or MalgraphClientProblem (False)
            threshold_type: '100fpr' or '1000fpr'
            device: torch.device or None (auto-detect)
            server_port: Port for remote IDA Pro API
        """
        self.threshold_type = threshold_type
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Initializing SRL-MalGraph Classifier Adapter")
        print(f"  Device: {self.device}")
        print(f"  Threshold: {threshold_type}")
        
        if use_direct_client:
            # Use DirectMalgraphClient (direct model loading, no TorchServe)
            print("  Mode: Direct client (no TorchServe)")
            self.classifier = DirectMalgraphClient(
                threshold_type=threshold_type,
                device=self.device,
                server_port=server_port
            )
            self.mode = 'direct'
        else:
            # Use MalgraphServerFeature (requires TorchServe)
            print("  Mode: Server feature (requires TorchServe)")
            self.classifier = self._load_server_feature(threshold_type)
            self.mode = 'server'
        
        self.threshold = self.classifier.clsf_threshold
        print(f"  Classification threshold: {self.threshold}")
        print("✓ Classifier adapter initialized\n")
    
    def _load_server_feature(self, threshold_type: str) -> MalgraphServerFeature:
        """Load MalgraphServerFeature (for TorchServe mode)."""
        # Load config
        base_path = str(Path(__file__).parents[2] / "GenAI_Malware_Repository/development_code/asm_pe_file_exp/MalGuise/MalGuise")
        cfg_path1 = os.path.join(base_path, 'configs/model_and_data.yaml')
        config1 = omegaconf.OmegaConf.load(cfg_path1)
        cfg_path2 = os.path.join(base_path, 'configs/preprocess.yaml')
        config2 = omegaconf.OmegaConf.load(cfg_path2)
        config = omegaconf.OmegaConf.merge(config1, config2)
        
        # Setup paths
        SCRIPT_PATH = os.path.join(base_path, config.Malgraph.Model.SCRIPT_PATH)
        tmp_sample_root = os.path.join(base_path, config.Malgraph.Model.tmp_sample_root)
        vocab_path = os.path.join(base_path, config.Malgraph.Model.vocab_path)
        model_path = os.path.join(base_path, config.Malgraph.Model.model_path)
        IDA_PATH = config.IDA_PATH
        acfg_dir_path = config.Malgraph.Model.extracted_cfg_dir
        
        # Create model parameters
        malgraph_model_params = MalgraphModelParams(
            gnn_type=config.Malgraph.Model.gnn_type,
            pool_type=config.Malgraph.Model.pool_type,
            acfg_init_dims=config.Malgraph.Model.acfg_init_dims,
            vocab_path=vocab_path,
            max_vocab_size=config.Malgraph.Model.max_vocab_size,
            cfg_filters=config.Malgraph.Model.cfg_filters,
            fcg_filters=config.Malgraph.Model.fcg_filters,
            skip_att_heads=config.Malgraph.Model.skip_att_heads,
            dropout_rate=config.Malgraph.Model.dropout_rate,
            ablation_models=config.Malgraph.Model.ablation_models,
            model_path=model_path,
            IDA_PATH=IDA_PATH,
            SCRIPT_PATH=SCRIPT_PATH,
            tmp_sample_root=tmp_sample_root,
            threshold_type=threshold_type,
            extracted_cfg_dir=acfg_dir_path
        )
        
        # Load model
        model = MalgraphServerFeature(malgraph_model_params, device=self.device)
        return model
    
    def predict(self, acfg_json) -> float:
        """
        Predict malware score for ACFG.
        
        THIS IS THE KEY METHOD FOR SRL ENVIRONMENT!
        
        Args:
            acfg_json: Either:
                - dict: ACFG dictionary with mutated block_features
                - str/Path: Path to JSON file containing ACFG data
        
        Returns:
            float: Malware confidence score (0-1)
        """
        import json
        
        # Handle file path input
        if isinstance(acfg_json, (str, Path)):
            with open(acfg_json, 'r') as f:
                data = json.load(f)
            
            # Extract the 'result' field if it exists (nested JSON string)
            if 'result' in data:
                acfg_json = json.loads(data['result'])
            else:
                acfg_json = data
        
        if self.mode == 'direct':
            # DirectMalgraphClient expects bytes, but we have ACFG JSON
            # We bypass get_score and call the model directly
            score = self.classifier.model(acfg_json)
            #print(f"DirectMalgraphClient score: {score}")
        else:
            # MalgraphServerFeature
            score = self.classifier(acfg_json)
        
        # Convert tensor to float if needed
        if torch.is_tensor(score):
            score = score.item()
        
        return float(score)
    
    def predict_from_binary(self, bytez: bytes, data_hash: str) -> tuple:
        """
        Predict malware score from binary (with ACFG extraction).
        
        This uses the full pipeline: binary → IDA Pro → ACFG → score
        
        Args:
            bytez: Binary bytes
            data_hash: Sample hash/name
        
        Returns:
            (score, status_code)
        """
        if self.mode == 'direct':
            score, status = self.classifier.get_score(bytez, data_hash)
            return score, status
        else:
            # Server mode
            score = self.classifier.get_score_custom(bytez, data_hash)
            return score, None
    
    def get_model(self):
        """
        Get the underlying MalGraph model for advanced use.
        
        Returns:
            The MalGraph model (HierarchicalGraphNeuralNetwork)
        """
        if self.mode == 'direct':
            return self.classifier.model.model  # DirectMalgraphClient → MalgraphServerFeature → model
        else:
            return self.classifier.model


# Example usage
if __name__ == "__main__":
    # Initialize adapter
    adapter = SRLMalGraphClassifierAdapter(
        use_direct_client=True,
        threshold_type='100fpr',
        device=None,  # Auto-detect
        server_port=5001
    )

    
    # Test 1b: Predict from ACFG JSON file
    print("\nTest 1b: Predict from ACFG JSON file")
    test_file = "/home/newdrive/makil/projects/GenAI_Malware_Repository/development_code/asm_pe_file_exp/MalGuise/MalGuise/src/utils/acfg_extractor/extracted_data/malgraph/MALGRAPH_BETA_3_300_CALIB_SET/c5e800da0e1d523119de112aef349fb586d37e35a589be9f95e2bad81b6d8798_September.json"
    try:
        score = adapter.predict(test_file)
        print(f"  Score: {score:.4f}")
        print(f"  Bypassed: {score < adapter.threshold}")
    except FileNotFoundError:
        print(f"  (Skipped - file not found)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Predict from binary (full pipeline)
    print("\nTest 2: Predict from binary (requires actual malware file)")
    # score, status = adapter.predict_from_binary(bytez, 'malware_hash')
    print("  (Skipped - needs actual binary)")
    
    print("\n✓ Adapter ready for SRL environment!")
