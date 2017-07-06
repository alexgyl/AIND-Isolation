"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    ## Game condition checks
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    ## Number of my moves vs exponential number of opponent moves heuristic
    # Punishes player more heavily if opponent has more moves
    num_player_moves = len(game.get_legal_moves(player))
    num_opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    score = float(num_player_moves - math.exp(num_opponent_moves))
    return score

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    ## Game condition checks
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    ## Comparing euclidean distance to the nearest corner of the board for player
    width = game.width
    height = game.height
    corners = [(0, 0), (0, width), (height, 0), (width, height)]
    # corner_1 = (0, 0)
    # corner_2 = (0, width)
    # corner_3 = (height, 0)
    # corner_4 = (width, height)
    nearest = float("inf")
    nearest_corner = None
    ## Player to closest corner
    for corner in corners:
        p_location = np.array(game.get_player_location(player))
        t_corner = np.array(corner)
        dist = np.linalg.norm(p_location - t_corner)
        if(dist < nearest):
            nearest = dist
            nearest_corner = corner

    ## Check opponents euclidean distance to nearest corner
    oppo_dist = np.linalg.norm(np.array(game.get_player_location(game.get_opponent(player))) - np.array(nearest_corner))
    # Compare distance - if player is further from the corner, we are in a better situation
    # Less likely to be cornered
    score = float(math.exp(nearest - oppo_dist))

    return score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    ## Game condition checks
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    ## Similar to custom score 1 but here we will use the # of blank spaces left
    # If player has more moves than its opponent, a game state where more blank spaces
    # are left should be worth LESS than one where less blank spaces are left.
    # This stems from the idea that I have MORE moves than you and there's less spaces
    # left, hence I should have a higher chance to win than you do.
    num_player_moves = len(game.get_legal_moves(player))
    num_opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    score = float(num_player_moves - (math.exp(num_opponent_moves/len(game.get_blank_spaces()))))
    return score

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        ## Check if there are any legal moves for current active player (game is
        ## the Board class) or if illegal search depth (depth 0) is input
        legal_moves = game.get_legal_moves()
        if not legal_moves or depth == 0:
            return (-1, -1)

        ## Perform the minimax algorithm here
        # From either player's POV, the next move will always be a minimizing move
        # First set the default value if no best move is ever returned & best score 
        best_move = (-1,-1)
        best_score = float("-inf")

        # At the top of the branch, cycle through all possible moves
        for current_move in legal_moves:
            # Call the min value function and decrease depth (we go one level down)
            score = self.minimizer(game.forecast_move(current_move), depth - 1)
            if score > best_score:
                best_score = score
                best_move = current_move
        return best_move


        ## Maximum value function
    def maximizer(self, game, depth):
        # Search timeout 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()           
        # Terminal depth - return the game score
        if depth == 0:
            return self.score(game, self)
        # No legal moves check - means that we are at the end of the game/branch
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return game.utility(self)
        # If all checks are passed, call the minimizer
        # First, set the worst possible value for the maximizer
        best_score = float("-inf")
        for current_move in legal_moves:
            # Call min value function and decrease depth (we go one level down)
            score = self.minimizer(game.forecast_move(current_move), depth - 1)
            if score > best_score:
                best_score = score
        return best_score

    ## Minimum value function
    def minimizer(self, game, depth):
        # Search timeout 
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()      
        # Terminal depth - return the game score
        if depth == 0:
            return self.score(game, self)
        # No legal moves check - means that we are at the end of the game/branch
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return game.utility(self)
        # If all checks are passed, call the maximizer
        # First, set the worst possible value for the minimizer
        best_score = float("inf")
        for current_move in legal_moves:
            # Call the max value function and decrease depth (we go one level down)
            score = self.maximizer(game.forecast_move(current_move), depth - 1)
            if score < best_score:
                best_score = score
        return best_score
        raise NotImplementedError

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        ## Current best move
        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return best_move
        ## Alpha-beta search
        try:
            # Initial depth
            max_depth = 0
            # Iterative deepening - max_depth tracks the maximum depth allowed
            while self.time_left() > self.TIMER_THRESHOLD:
                best_move = self.alphabeta(game, max_depth)
                max_depth += 1
        except SearchTimeout:
            return best_move

        return best_move
        # TODO: finish this function!
        raise NotImplementedError


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!

        ## Maximizing function with alpha-beta pruning
        def maximizer(self, game, depth, alpha, beta):
            ## Timeout check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            ## No legal moves left
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return game.utility(self)
            ## Depth check
            if depth == 0:
                return self.score(game, self)
            ## Maximizer with alpha-beta considerations
            # Worst possible score for maximizer
            score = float("-inf")
            for current_move in legal_moves:
                score = max(score, minimizer(self, game.forecast_move(current_move), depth - 1, alpha, beta))
                # Alpha-beta pruning conditions
                if score >= beta:
                    return score 
                alpha = max(alpha, score)
            return score

        ## Minimizing function with alpha-beta pruning
        def minimizer(self, game, depth, alpha, beta):
            ## Timeout check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()        
            ## No legal moves left
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return game.utility(self)
            ## Depth check
            if depth == 0:
                return self.score(game, self)
            ## Minimizer with alpha-beta considerations
            # Worst possible score for minimizer
            score = float("inf")
            for current_move in legal_moves:
                score = min(score, maximizer(self, game.forecast_move(current_move), depth - 1, alpha, beta))
                # Alpha-beta pruning conditions
                if score <= alpha:
                    return score 
                beta = min(beta, score)
            return score 
        # --------------- Main body of alpha-beta -----------------------------    
        ## Check for legal moves or valid depth
        legal_moves = game.get_legal_moves()
        ## Max depth check
        if depth == 0:
            return self.score(game, self)
        ## No legal moves left
        if not legal_moves:
            return game.utility(self)
        ## Perform alpha-beta search here once checks are done
        ## Setting the best score for maxmizer & worst cases for alpha-beta
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        best_move = None
        # Perform the search for all legal moves
        for current_move in legal_moves:
            score = minimizer(self, game.forecast_move(current_move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = current_move
            alpha = max(alpha, score)
        ## Returning the best move possible
        return best_move
        
        raise NotImplementedError
