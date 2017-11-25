"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import copy

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def modified_center_score(game, player):
    """
        Outputs a score equal to manhattan distance of the current position of of the player from
        the center of the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height / 2., game.width / 2.
    y, x = game.get_player_location(player)
    return abs(y - h) + abs(x - w)
    #return float(h ** 2 + w ** 2) - float((h - y) ** 2 + (w - x) ** 2)

def improved_score(game, player):
    """The "Improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def move_is_legal(game, move, blank_locations):
    '''
        Checks whether the given move is legal in light of the game configuration and blank squares
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    move: tuple
        A tuple in the form of (row, column) which identifies the next move on the board for a player
    blank_locations: dict
        A dict that maps tuples (row, column) to a boolean indicating whether the positon (row, column) on the
        board is still empty
    
    Returns
    _______
    
    boolean
        True if the given move is legal else False
    '''
    return (0 <= move[0] < game.height and 0 <= move[1] < game.width and
            move in blank_locations)

def get_moves(game, loc, blank_locations):
    '''
        Finds all legal moves possible from given position on the board and returns them
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    loc: tuple
        A tuple in the form of (row, column) which identifies the current position on the board for a player
    blank_locations: dict
        A dict that maps tuples (row, column) to a boolean indicating whether the positon (row, column) on the
        board is still empty
    
    Returns
    _______
    
    list 
        A list of tuples of the form (row, column) identifying the legal moves that can be made from given position 
    '''
    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if move_is_legal(game, (r + dr, c + dc), blank_locations)]
    return valid_moves

def find_max_chain_length(game, player_loc, blank_locations, depth, memo):
    '''
        Finds the length of maximum chain of moves possible from the current position of the player on the board
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player_loc: tuple
        A tuple in the form of (row, column) which identifies the current position on the board for a player
    blank_locations: dict
        A dict that maps tuples (row, column) to a boolean indicating whether the positon (row, column) on the
        board is still empty 
    depth: int
        An int indicating the current level
    memo: dict
        A dict which stores maximum chain length for chains starting from a particular position.
        Key: (row, column) - indicating a position in the board
        Value: int - maximum chain length for a chain beginning from (row, column)
    
    Returns
    _______
    
    int
        An int representing the maximum chain length of moves possible from given player_loc
    '''
    # It doesn't help much to look beyond 7 moves or more since the future rewards are hard to predict beyond a particular level
    # Also, there is diminishing returns and a higher chance of timeouts the deeper we explore
    if depth > 7:
        return 0
    # Get legal moves from player_loc
    legal_moves = get_moves(game, player_loc, blank_locations)
    # If result is already available, then return it
    if player_loc in memo.keys():
        return memo[player_loc]
    max_chain_length = -1
    for move in legal_moves:
        blank_locations_copy = copy.deepcopy(blank_locations)
        del blank_locations_copy[move]
        # Computer maximum chain length from player_loc recursively
        max_chain_length = max(max_chain_length, find_max_chain_length(game, move, blank_locations_copy, depth + 1, memo) + 1)
    # Memoize the result for player_loc in the dictionary
    memo[player_loc] = max_chain_length
    return max_chain_length

def find_closure_length(game, player_loc, blank_locations, depth = 2):
    '''
        Finds the number of unique moves possible from the current position of the player on the board
        limited by given depth

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player_loc: tuple
        A tuple in the form of (row, column) which identifies the current position on the board for a player
    blank_locations: dict
        A dict that maps tuples (row, column) to a boolean indicating whether the positon (row, column) on the
        board is still empty 
    depth: int
        An int indicating the maximum depth for computing closure

    Returns
    _______

    int
        An int representing the number of unique moves in the closure from given player_loc
    '''
    closure_moves = {}
    closure_moves.update([(move, True) for move in get_moves(game, player_loc, blank_locations)])

    for i in range(0, depth):
        closure_moves.update([(next_move, True) for move in closure_moves for next_move in get_moves(game, move, blank_locations)])

    return len(closure_moves)

def list_to_dict(l):
    # Helper method to convert a list to a dictionary with Key as element and Value as 1
    return dict([(item, 1) for item in l])

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function combines two heuristics namely - the modified center score
    and the difference in lengths of moves closure.
    
    The length of moves closure is nothing but the number of unique positions that are reachable
    from current position of the player on the board in subsequent turns.
    
    The difference between length of current player's move closure and length of opponent player's move closure
    is the heuristic value. It is a reasonably good indicator of the win probability of the current player.
    
    The squares near the center of the board are highly desirable in the early stages of the game since the 
    branching factor is quite high near the center squares. For this reason, this heuristic calculates 
    modified center score if number of moves is lower than 10, otherwise it uses length of moves closure.
    This naturally has the effect of forcing the agent to play near the center of the board in the opening few moves.

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
    if game.move_count < 10:
        return modified_center_score(game, player)
    else:
        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        current_player_sum = find_closure_length(game,
                                                 game.get_player_location(player),
                                                 list_to_dict(game.get_blank_spaces()))
        opponent_player_sum = find_closure_length(game,
                                                  game.get_player_location(game.get_opponent(player)),
                                                  list_to_dict(game.get_blank_spaces()))
        return float(current_player_sum - opponent_player_sum)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        The heuristic value is calculated using the modified center score

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
    return modified_center_score(game, player)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    This heuristic uses the difference between maximum length chain of moves possible for the given player 
    and the maximum length chain of moves possible for the opponent.

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
    # If the number of moves is less than 5, use modified center score
    if game.move_count < 5:
        return modified_center_score(game, player)
    # If the fill ratio is less than 40%, use improved score
    elif (game.move_count * 1.0 / (game.width * game.height)) < 0.4:
        return improved_score(game, player)
    else:
        memo = {}
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("inf")
        current_player_max_chain = find_max_chain_length(game,
                                                   game.get_player_location(player),
                                                   list_to_dict(game.get_blank_spaces()),
                                                   0,
                                                   memo)
        opponent_player_max_chain = find_max_chain_length(game,
                                                    game.get_player_location(game.get_opponent(player)),
                                                    list_to_dict(game.get_blank_spaces()),
                                                    0,
                                                    memo)
        return float(current_player_max_chain - opponent_player_max_chain)


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
        """

        best_move = (-1, -1)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Calculate the available legal moves on board for this player
        legal_moves = game.get_legal_moves(self)

        best_minimax_value = float("-inf")
        try:
            # Iterate over the list of legal moves in an arbitrary order
            for move in legal_moves:
                game_one_move_ahead = game.forecast_move(move)
                # Calculate minimax value for this new game state
                m = self.minimax_value(game_one_move_ahead, 1, depth, False)
                # If minimax value for this move is greater than best value so far, set this move as best move
                if m > best_minimax_value:
                    best_minimax_value = m
                    best_move = move
            return best_move
        except SearchTimeout:
            pass

        """
            This might happen due to search timeout ie. the algorithm might not have picked a legal move 
            eventhough there was one available. In such a case, to prevent from forfeiting the game due to timeout, 
            pick a legal move randomly.
        """
        if not game.move_is_legal(best_move):
            legal_moves = game.get_legal_moves(self)
            if legal_moves:
                return legal_moves[random.randrange(0, len(legal_moves))]
        return best_move

    def minimax_value(self, game, current_depth, max_depth, is_maximising):
        """
            Calculate minimax value of the current game state from the perspective of this player by recursive evaluation.
            When current_depth equals max_depth, we have hit the limit, so all leaf moves in the search tree
            will be scored using self.score() which in turn uses the configured heuristic functions to calculate
            utility value of the states from the perspective of this player
            
            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state
            
            current_depth: int
                The current depth of the minimax search tree
                
            is_maximising: boolean 
                Indicates whether the current level is maximising or minimising
            
            Returns
            ------- 
            float
                The minimax value of the current game state as seen by this player.
        """
        # Raise SearchTimeout exception if time left is lesser than configured threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Calculate the available legal moves on board for the currently active player
        legal_moves = game.get_legal_moves(game.active_player)

        if not legal_moves or current_depth >= max_depth:
            return self.score(game, self)

        # Iterate over the list of legal moves in an arbitrary order
        children_values = []
        # Calculate for every legal move the minimax value for game state after making that move
        for move in legal_moves:
            game_one_move_ahead = game.forecast_move(move)
            child_minimax_value = self.minimax_value(game_one_move_ahead, current_depth + 1, max_depth, not is_maximising)
            children_values.append(child_minimax_value)

        if is_maximising:
            return max(children_values)
        else:
            return min(children_values)

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

        best_move = (-1, -1)

        try:
            i = 1
            while True:
                best_move = self.alphabeta(game, i)
                i += 1
        except SearchTimeout:
            pass

        """
        This might happen due to search timeout ie. the algorithm might not have picked a legal move 
        eventhough there was one available. In such a case, to prevent from forfeiting the game due to timeout, 
        pick a legal move randomly.
        """
        if not game.move_is_legal(best_move):
            legal_moves = game.get_legal_moves(self)
            if legal_moves:
                return legal_moves[random.randrange(0, len(legal_moves))]
        return best_move

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
            lookahead in the game tree before aborting

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
        # Raise SearchTimeout exception if time left is lesser than configured threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)
        best_minimax_value = float("-inf")
        try:
            # Calculate the available legal moves on board for this player
            legal_moves = game.get_legal_moves(self)
            for move in legal_moves:
                # If alpha >= beta, prune this subtree
                if alpha >= beta:
                    break
                game_one_move_ahead = game.forecast_move(move)
                m = self.minimax_value_with_pruning(game_one_move_ahead, 1, depth, alpha, beta, False)
                if m > best_minimax_value:
                    best_move, best_minimax_value, alpha = move, m, m
        except SearchTimeout:
            raise
        return best_move

    def minimax_value_with_pruning(self, game, current_depth, max_depth, alpha, beta, is_maximising):
        """
        Calculate minimax value of the current game state from the perspective of this player by recursive evaluation.
        When current_depth equals max_depth, we have hit the limit, so all leaf moves in the search tree
        will be scored using self.score() which in turn uses the configured heuristic functions to calculate
        utility value of the states from the perspective of this player.
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        
        current_depth: int
            The current depth of the minimax search tree
        
        is_maximising: boolean 
            Indicates whether the current level is maximising or minimising
        
        Returns
        -------
        float
            The minimax value of the current game state as seen by this player.
        """

        # Raise SearchTimeout exception if time left is lesser than configured threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Calculate the available legal moves on board for the currently active player
        legal_moves = game.get_legal_moves(game.active_player)

        # If maximum search depth is reached or there are no legal moves, then simply call score to get the appropriate utility value
        if not legal_moves or current_depth >= max_depth:
            return self.score(game, self)

        init_value = lambda x: float("-inf") if x else float("inf")
        should_update = lambda x, y, z: y > z if x else y < z
        best_value = init_value(is_maximising)
        # Iterate over the list of legal moves in an arbitrary order
        # Calculate for every legal move the minimax value for game state after making that move
        for move in legal_moves:
            # If alpha >= beta, prune the search tree
            if alpha >= beta:
                break
            game_one_move_ahead = game.forecast_move(move)
            m = \
                self.minimax_value_with_pruning(game_one_move_ahead,
                                                current_depth + 1,
                                                max_depth,
                                                alpha,
                                                beta,
                                                not is_maximising)
            if should_update(is_maximising, m, best_value):
                best_value = m
                if is_maximising:
                    alpha, beta = max(alpha, m), beta
                else:
                    alpha, beta = alpha, min(beta, m)

        return best_value

