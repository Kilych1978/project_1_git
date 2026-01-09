print(f'Здравствуйте, детишки!\n\nСыграем в игру?')

open = input('Y/N?\n')

if open == 'n' or open == 'N':
    print('\nНу и пошёл на хуй!')
    exit()

print('\nСейчас я напишу сказочку, а вы допишите её\n')

def the_fable():
    open = input('У попа была собака, он её - ...')
    if open != 'любил':
        print('\nТы долбаёб что ли?\nДавай заново\n')
        return the_fable()
    print('Молодец!\n')

    open = input('Она съела кусок мяса, он её - ...')
    if open != 'убил':
        print('\nТы вообще блядь ёбнутый что ли?\nДавай заново\n')
        return the_fable()
    print('Умница!\n')

    open = input('В яму - ...')
    if open != 'закопал':
        print('\nТы реально долбаёб?\nДавай заново\n')
        return the_fable()
    print('Молорик!\n')

    open = input('Надпись - ...')
    if open != 'написал':
        print('\nНу ты и кончелыга!\nДавай заново\n')
        return the_fable()
    print('Молоток!\n')
    print('ЧТО:\n')

    the_fable()
    
the_fable()    
    