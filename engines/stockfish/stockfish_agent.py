"""
Wrapper for stockfish engine agent
"""
import chess.engine
class StockfishAgent:
    """
    Stockfish agent
    """
    def __init__(self, stockfish_path="stockfish_exe\stockfish-windows-x86-64-avx2.exe", level=None):
        """
        Get the stockfish executable and set level
        """
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.level = level
        
    def take_turn(self, board):
        pass