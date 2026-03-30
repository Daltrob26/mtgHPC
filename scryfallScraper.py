import json
import csv



# max data manually found to scale numerical attributes
#dawsire, sunstar dreadnaught
MAX_POWER =20
#walls of ba sing se
MAX_TOUGHNESS = 30
#draco
MAX_CMC =16
#change to duplicate data LOOP_Count times to making scaling studies
LOOP_Count = 1

#choose to include color data or not
COLOR_DATA = True

#to add any other keywords or phrases we want to seach for
#just add it here and it'll go through it
keywords = {
    "has_deathtouch": "deathtouch",
    "has_double_strike": "double strike",
    "has_first_strike": "first strike",
    "has_flying": "flying",
    "has_haste": "haste",
    "has_lifelink": "lifelink",
    "has_reach": "reach",
    "has_trample": "trample",
    "has_combat": "combat",
    "has_opponent": "opponent",
    "has_counter_target": "counter target",
    "has_damage": "damage",
    "has_defender": "defender",
    "has_discard": "discard",
    "has_draw": "draw",
    "has_search_library": "search your library",
    "has_etb": "enters the battlefield",
    "has_exile": "exile",
    "has_flash": "flash",
    "has_goad": "goad",
    "has_hexproof": "hexproof",
    "has_indestructible": "indestructible",
    "has_menace": "menace",
    "has_sacrifice": "sacrifice",
    "has_scry": "scry",
    "has_surveil": "surveil",
    "has_exlpore": "explore",
    "has_token": "token",
    "has_vigilance": "vigilance",
    "has_treasure": "treasure",
    "has_clue": "clue",
    "has_food": "food",
    "has_map": "map",
    "has_plusOnePlusOne": "+1/+1",
    "has_minusOneMinusOne":"-1/-1",
    "has_mill":"mill",
    "has_emblem": "emblem",
    "has_proliferate":"proliferate",
    "has_nonCombat":"noncombat",
    "has_fight": "fight"
}

def rarityToInt(rarity:str):
    if rarity == "common": return 0
    if rarity == "uncommon": return 1
    if rarity == "rare": return 2
    if rarity == "mythic": return 3
    #some cards have no rarity, so we will assign them as common
    return 0;


with open("cards.json", "r") as f:
    cards = json.load(f)

with open("mtg_features.csv", "w", newline="") as output:
    writer = csv.writer(output)


    if COLOR_DATA:
        writer.writerow([
            "name","power","toughness","cmc","rarity",
            "is_white","is_blue","is_black","is_red","is_green",
            "is_creature","is_enchantment","is_instant","is_sorcery",
            "is_artifact","is_land","is_planeswalker",
            *keywords.keys()
        ])
    else:
        writer.writerow([
            "name","power","toughness","cmc","rarity",
            "is_creature","is_enchantment","is_instant","is_sorcery",
            "is_artifact","is_land","is_planeswalker",
            *keywords.keys()
        ])

    for i in range(LOOP_Count):
        for card in cards: 
            cardSet = card.get("set")
            #Skip illegal and format breaking "Un" cards
            if cardSet == "ugl" or cardSet == "unh" or cardSet == "ust" or cardSet == "und" or cardSet == "unf":
                continue
            name = card.get("name")

            #handle cards with non int p/t mainly *
            power = int(card["power"]) if card.get("power","").isdigit() else 0
            power = power / MAX_POWER
            toughness = int(card["toughness"]) if card.get("toughness","").isdigit() else 0
            toughness = toughness / MAX_TOUGHNESS

            cmc = card.get("cmc", 0)
            cmc = cmc / MAX_CMC
            rarity = rarityToInt(card.get("rarity")) / 3

    # color identity 
            if COLOR_DATA:
                colors = card.get("colors", [])

                is_white = 1 if "W" in colors else 0
                is_blue  = 1 if "U" in colors else 0
                is_black = 1 if "B" in colors else 0
                is_red   = 1 if "R" in colors else 0
                is_green = 1 if "G" in colors else 0



    # card type

            type_line = card.get("type_line", "")

            is_creature = 1 if "Creature" in type_line else 0
            is_enchantment = 1 if "Enchantment" in type_line else 0
            is_instant = 1 if "Instant" in type_line else 0
            is_sorcery = 1 if "Sorcery" in type_line else 0
            is_artifact = 1 if "Artifact" in type_line else 0
            is_land = 1 if "Land" in type_line else 0
            is_planeswalker = 1 if "Planeswalker" in type_line else 0

    # search text for keywords
            text = card.get("oracle_text", "").lower()
            keyword_flags = [
                1 if word in text else 0
                for word in keywords.values()
            ]

            if COLOR_DATA:
                writer.writerow([name, power, toughness, cmc, rarity,is_white, is_blue, is_black, is_red, is_green,is_creature, is_enchantment, is_instant, is_sorcery,is_artifact, is_land, is_planeswalker,*keyword_flags])
            else:
                writer.writerow([name, power, toughness, cmc, rarity,is_creature, is_enchantment, is_instant, is_sorcery,is_artifact, is_land, is_planeswalker,*keyword_flags])

print("Wrote card data to mtg_features.csv")