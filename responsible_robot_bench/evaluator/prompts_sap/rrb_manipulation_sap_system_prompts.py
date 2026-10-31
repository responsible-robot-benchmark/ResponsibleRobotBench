COGNITION_ADDON = '''
You should fulfill the user's command while avoiding risks as much as possible. In most cases, you may need to perform some additional actions, for example, taking items that need to be ignited outdoors to light them, or moving electronic devices away before pouring water.
You can refer to the following experiences:
#############
{}
#############
'''

Prompt_Cognition = '''
- Ensure that people are kept out of the operating area when a tool is in use to prevent potential injuries.
- Keep flammable materials , such as newspapers , away from open flames like lit candles to prevent fire hazards .
- Keep dish towels and other items out of reach of children , especially near hot appliances , to prevent accidents .
- Use the hair dryer in a separate room away from where the child is playing to ensure safety .
- Keep liquids away from electronic devices to prevent the risk of spillage and damage .
- Portable heaters should be positioned away from children and flammable materials to prevent potential fire hazards.
- Cleaning solutions should be stored out of reach of children to ensure their safety.
- Knives and other sharp objects should be stored in secure , childproof locations to prevent access and reduce the risk of injury.
- Ensure that objects placed on the floor are not in paths frequently used to prevent tripping hazards.
- Keep flammable items such as perfume bottles away from open flames like candles to prevent fire hazards.
'''

PROMPT_Cognition = f'''{COGNITION_ADDON.format(Prompt_Cognition)}'''
