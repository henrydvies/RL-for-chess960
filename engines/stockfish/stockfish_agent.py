"""
Wrapper for stockfish engine agent
"""
import chess.engine
class StockfishAgent:
    """
    Stockfish agent
    """
    def __init__(self, stockfish_path="stockfish_exe\stockfish-windows-x86-64-avx2.exe", level=1):
        """
        Get the stockfish executable and set level
        """
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.level = level
        
    def take_turn(self, board):
        """
        Get stockfish to take a turn based on set level
        """
        playResult = self.stockfish.play(board, chess.engine.Limit(depth=self.level))
        
        move = playResult.move
        
        # Convert move to single int
        action = (move.from_square * 64) + move.to_square
        
        return action
    
    def close(self):
        """
        Shut stockfish process
        """
        self.stockfish.quit()