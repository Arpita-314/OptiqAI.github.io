"""
File Manager for Black-Scholes Calculations
Handles saving/loading calculations, history, and data persistence
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle


class FileManager:
    """
    Manages file operations for Black-Scholes calculations
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize file manager
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.calculations_dir = self.data_dir / "calculations"
        self.history_dir = self.data_dir / "history"
        self.export_dir = self.data_dir / "exports"
        
        for dir_path in [self.calculations_dir, self.history_dir, self.export_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_calculation(self, calculation_id: str, data: Dict[str, Any]) -> str:
        """
        Save a calculation to file
        
        Args:
            calculation_id: Unique identifier for the calculation
            data: Calculation data to save
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{calculation_id}_{timestamp}.json"
        filepath = self.calculations_dir / filename
        
        # Add metadata
        data_with_meta = {
            "calculation_id": calculation_id,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "data": data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_with_meta, f, indent=2)
        
        return str(filepath)
    
    def load_calculation(self, calculation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a calculation by ID
        
        Args:
            calculation_id: Calculation identifier
            
        Returns:
            Calculation data or None if not found
        """
        # Find most recent file with this ID
        pattern = f"{calculation_id}_*.json"
        files = list(self.calculations_dir.glob(pattern))
        
        if not files:
            return None
        
        # Get most recent
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def save_conversation_history(self, session_id: str, messages: List[Dict]) -> str:
        """
        Save conversation history
        
        Args:
            session_id: Session identifier
            messages: List of conversation messages
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp}.json"
        filepath = self.history_dir / filename
        
        data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def load_conversation_history(self, session_id: str) -> Optional[List[Dict]]:
        """
        Load conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages or None
        """
        pattern = f"session_{session_id}_*.json"
        files = list(self.history_dir.glob(pattern))
        
        if not files:
            return None
        
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
            return data.get("messages", [])
    
    def export_to_csv(self, calculations: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Export calculations to CSV
        
        Args:
            calculations: List of calculation dictionaries
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported CSV file
        """
        if not calculations:
            raise ValueError("No calculations to export")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calculations_export_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        # Flatten nested dictionaries for CSV
        rows = []
        for calc in calculations:
            row = {}
            if isinstance(calc, dict):
                if "data" in calc:
                    # Handle our saved format
                    row.update(calc.get("data", {}))
                    row["calculation_id"] = calc.get("calculation_id", "")
                    row["timestamp"] = calc.get("timestamp", "")
                else:
                    row.update(calc)
            rows.append(row)
        
        if not rows:
            raise ValueError("No valid data to export")
        
        # Get all unique keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        
        fieldnames = sorted(list(all_keys))
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        return str(filepath)
    
    def list_calculations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent calculations
        
        Args:
            limit: Maximum number of calculations to return
            
        Returns:
            List of calculation metadata
        """
        files = sorted(
            self.calculations_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        calculations = []
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    calculations.append({
                        "file": str(filepath.name),
                        "calculation_id": data.get("calculation_id", ""),
                        "timestamp": data.get("datetime", ""),
                        "summary": self._extract_summary(data.get("data", {}))
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        return calculations
    
    def _extract_summary(self, data: Dict) -> str:
        """Extract a summary string from calculation data"""
        if "price" in data:
            return f"Price: ${data['price']:.2f}"
        elif "option_price" in data:
            return f"Option Price: ${data['option_price']:.2f}"
        elif "implied_volatility" in data:
            return f"IV: {data['implied_volatility']*100:.2f}%"
        else:
            return "Calculation"
    
    def delete_calculation(self, calculation_id: str) -> bool:
        """
        Delete calculations by ID
        
        Args:
            calculation_id: Calculation identifier
            
        Returns:
            True if deleted, False if not found
        """
        pattern = f"{calculation_id}_*.json"
        files = list(self.calculations_dir.glob(pattern))
        
        if not files:
            return False
        
        for filepath in files:
            try:
                filepath.unlink()
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored data
        
        Returns:
            Dictionary with statistics
        """
        calculation_files = list(self.calculations_dir.glob("*.json"))
        history_files = list(self.history_dir.glob("*.json"))
        export_files = list(self.export_dir.glob("*.csv"))
        
        total_size = sum(
            f.stat().st_size for f in calculation_files + history_files + export_files
        )
        
        return {
            "total_calculations": len(calculation_files),
            "total_sessions": len(history_files),
            "total_exports": len(export_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_directory": str(self.data_dir)
        }

