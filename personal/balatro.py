import eval7, pprint
import numpy as np
import matplotlib.pyplot as plt

class PokerHands:

    RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
    SUITS = ('c', 'd', 'h', 's')
    HAND_SIZE = 5

    def __init__(self, hand, deck, starting_hand_size=8, verbose = True):

        self.starting_hand_size=starting_hand_size
        self.deck=deck
        self.hand=[]
        self.obs_suits = {} # history of all observed suits
        self.obs_ranks = {} # history of all observed ranks

        self.min_suit_count = self.starting_hand_size // 4 # absolute minimum num of suits that is possible to have in hand for intended flush suit
        self.update(hand)
        
        # Settings
        self.verbose = verbose

    def get_counts(self, x_list, count_dict={}):
        """Get count of elements in a list"""
        for x in x_list:
            count_dict[x] = 1 + count_dict.get(x,0)
        return count_dict

    def get_suit_count(self, hand, count_dict={}):
        """Counts the number of suits, and updates if adding to an existing count dict"""
        current_suit_count = self.get_counts(
            [self.SUITS[h.suit] for h in hand], 
            count_dict)
        return current_suit_count

    def get_rank_count(self, hand, count_dict={}):
        """Counts the number of ranks, and updates if adding to an existing count dict"""
        current_rank_count = self.get_counts(
            [self.RANKS[h.rank] for h in hand], 
            count_dict)
        return current_rank_count

    def update(self, hand):
        """Update hand, historical observed suits, and observed historical ranks"""
        self.hand += hand 
        self.obs_suits = self.get_suit_count(hand, self.obs_suits)
        self.obs_ranks = self.get_rank_count(hand, self.obs_ranks)
    
    def is_flush(self, current_suits=None):
        """Detects if hand has a flush"""
        if not current_suits:
            current_suits=self.obs_suits
        has_flush = any(count >= 5 for count in current_suits.values())
        return has_flush
    
    def is_straight(self, current_ranks=None):
        """Detects if hand has a straight"""
        if not current_ranks:
            current_ranks=self.obs_ranks
        # All possible straights
        straight_references = [set(['A','2', '3', '4', '5'])] + [set(self.RANKS[i:i+5]) for i in range(9)]

        hand_set = set(current_ranks.keys())

        straight_mask = [straight.issubset(hand_set) for straight in straight_references]

        has_straight = any(straight_mask)
        return has_straight


    def simulate_one_flush_round(self):
        """Simulate one round with the goal of getting a flush"""
        # Method only goes for the most abundant suit in hand, not what is most likely to be drawn

        num_discards = 0
        current_suits = self.obs_suits
        while not self.is_flush(current_suits):
            # Deciding which suit to keep
            max_count = max(current_suits.values())
            most_common_suits = [suit for suit, count in current_suits.items() if count == max_count]

            # Removing cards and then updating
            suit_to_keep = np.random.choice(most_common_suits).item() # randomly pick
            hand_to_keep = [
                card for i, card in enumerate(self.hand) 
                if self.SUITS[card.suit]==suit_to_keep
                ] # filter to keep only target suit
            if len(hand_to_keep) <= self.min_suit_count:
                # if most common suit has 2 counts, then we need to keep 1 more card to satisfy hand size
                # only happens if suit count is 2-2-2-2 and hand_size is 8
                random_card = next(card for card in self.hand if self.SUITS[card.suit]!=suit_to_keep)
                hand_to_keep += [random_card]
            self.hand = hand_to_keep

            cards_to_discard = self.starting_hand_size - len(self.hand)
            discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
            if discarded_more_than_hand_size:
                raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards, but ended up discarding {cards_to_discard} cards")
            
            try:
                new_cards = self.deck.deal(cards_to_discard)
            except ValueError:
                if self.verbose:
                    print("Ran out of cards to draw")
                    num_discards = None
                    return num_discards
            self.update(new_cards)
            current_suits = self.get_suit_count(self.hand, count_dict={})

            num_discards += 1

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a flush")
        return num_discards

    def simulate_one_straight_round(self):
        """Simulate one round with the goal of getting a straight"""
        # Does not value higher straights
        straight_references = [set(['A','2', '3', '4', '5'])] + [set(self.RANKS[i:i+5]) for i in range(9)]

        num_discards = 0
        current_ranks = self.obs_ranks
        while not self.is_straight(current_ranks):
            # Detect which straight is most likely
            hand_rank_set = set(current_ranks.keys())
            straight_references_hits = [len(s.intersection(hand_rank_set)) for s in straight_references]
            target_straight_ind = np.argmax(straight_references_hits[::-1])
            target_straight = straight_references[-(target_straight_ind+1)] # target the highest viable straight

            # keep only the values in the target straight
            hand_to_keep = [
                card for i, card in enumerate(self.hand) 
                if self.RANKS[card.rank] in target_straight
                ] # filter to only keep ranks in target straight
            current_ranks = self.get_rank_count(hand_to_keep, count_dict={}) # get rank count to check for duplicates
            max_rank_count = max(current_ranks.values())
            hand_to_keep_has_duplicates = max_rank_count > 1
            if hand_to_keep_has_duplicates:
                # for each duplicate rank, pick a random card
                duplicate_ranks_in_hand_to_keep = [rank for rank, count in current_ranks.items() if count == max_rank_count] # identified duplicate ranks
                duplicate_rank_to_keep = [
                    next(card for card in hand_to_keep if self.RANKS[card.rank] == rank) # picking one card of a given rank 
                    for rank in duplicate_ranks_in_hand_to_keep
                    ]
                # recompose hand to keep as unique ranks + deduplicated ranks
                hand_to_keep = [card for card in hand_to_keep if self.RANKS[card.rank] not in duplicate_ranks_in_hand_to_keep] + duplicate_rank_to_keep
            
            cards_to_discard = self.starting_hand_size - len(hand_to_keep)
            discarding_too_many_cards = cards_to_discard > self.HAND_SIZE
            if discarding_too_many_cards:
                # need to add a random card back
                #TODO: what happens if I need to grab more than 1 card?
                random_card = next(card for card in self.hand if card not in hand_to_keep)
                hand_to_keep.append(random_card)

            self.hand = hand_to_keep
            cards_to_discard = self.starting_hand_size - len(self.hand)
            discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
            if discarded_more_than_hand_size:
                raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards is not allowed. Ended up discarding {cards_to_discard} cards")

            try:
                new_cards = self.deck.deal(cards_to_discard)
            except ValueError:
                if self.verbose:
                    print("Ran out of cards to draw")
                    num_discards = None
                    return num_discards
            self.update(new_cards)
            current_suits = self.get_rank_count(self.hand, count_dict={})
            num_discards += 1

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a straight")
        return num_discards


    def reset(self):
        self.hand = []
        self.obs_suits = {}
        self.obs_ranks = {}

def main():
    starting_hand_size = 8

    deck = eval7.Deck()
    deck.shuffle()
    starting_hand = deck.deal(starting_hand_size)
    PH = PokerHands(starting_hand, deck, starting_hand_size)
    #print(PH.simulate_one_flush_round())
    print(PH.simulate_one_straight_round())
    print()

def simulation(method, num_simulations=1000, verbose=False):


    starting_hand_size = 8
    list_of_discards = []
    for _ in range(num_simulations):
        deck = eval7.Deck()
        deck.shuffle()
        starting_hand = deck.deal(starting_hand_size)
        PH = PokerHands(starting_hand, deck, starting_hand_size, verbose=verbose)


        if method == "flush":
            run_sim = PH.simulate_one_flush_round
        elif method == "straight":
            run_sim = PH.simulate_one_straight_round
        else:
            raise ValueError("Unexpected method")
        list_of_discards.append(run_sim())
    
    return list_of_discards

if __name__ == "__main__":
    # main()

    num_simulations = 100_000
    flush_discards = simulation("flush", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got flushes with {np.mean(flush_discards)} discards on average")
    print(f"\tThere were {sum([1 if v == None else 0 for v in flush_discards])} rounds that ran out of cards in the deck")

    straight_discards = simulation("straight", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got straights with {np.mean(straight_discards)} discards on average")
    print(f"\tThere were {sum([1 if v == None else 0 for v in straight_discards])} rounds that ran out of cards in the deck")

    fig, axs = plt.subplots(1, 2)

    axs[0].hist(flush_discards)
    axs[0].set_title("Num discards for Flushes")
    axs[1].hist(straight_discards)
    axs[1].set_title("Num discards for Straights")
    plt.show()

    print()