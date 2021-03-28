
import os
from collections import namedtuple


# i. 인풋인자인 fileName 예시: ~~/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png /21.3.28.9:52. 
def getCsFileInfo(fileName):
    """Returns a CsFile object filled from the info in the given filename"""
    baseName = os.path.basename(fileName) # i. ex: frankfurt_000000_000294_gtFine_labelIds.png /21.3.28.9:53. 
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError('Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName))
    if len(parts) == 5:
        csFile = CsFile(*parts[:-1], type2="", ext=parts[-1])
    elif len(parts) == 6:
        csFile = CsFile(*parts)
    else:
        printError('Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts), fileName))



fileName = "some_rootJ/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png"
baseName = os.path.basename(fileName) # i. ex: frankfurt_000000_000294_gtFine_labelIds.png /21.3.28.9:53. 
print(f'j) baseName: {baseName}')
parts = baseName.split('_')
print(f'j) parts: {parts}')
parts = parts[:-1] + parts[-1].split('.')
print(f'j) parts: {parts}') 




# i.21.3.28.10:42) 파이썬 collections 의 namedtuple 실습. 
# StudentNT = namedtuple('studentJ', ['nameJ', 'ageJ', 'locationJ'])
# print(StudentNT)
# s1 = StudentNT('juny', '5555', 'SanFrancisco')
# s2 = StudentNT('suny', '5547', 'SanFrancisco')
# print(StudentNT)
# print(s1)
# print(s2)