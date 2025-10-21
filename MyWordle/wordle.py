# region Importation des bibliothèques nécessaires
import tkinter as tk
import random 
# endregion

# region Comptage du nombre d'occurrences d'un élément dans une liste
def count(lst, element):
    compteur = 0
    for item in lst:
        if item == element:
            compteur += 1
    return compteur
#endregion

# region Recherche des lettres correctes et de leurs positions
def comparaison(mot1, mot2):
    lettres_bonnes_position=[]
    lettres_mauvaises_position=[]
    for i in range(len(mot1)):
        if mot1[i]==mot2[i]:
            lettres_bonnes_position.append(i)
    for i in range(len(mot1)):
        if mot1[i] in mot2 and i not in lettres_bonnes_position:
            if count([mot2[i] for i in lettres_bonnes_position],mot1[i]) + count([mot2[i] for i in lettres_mauvaises_position],mot1[i]) < count(mot2,mot1[i]):
                lettres_mauvaises_position.append(i)
    if len(lettres_bonnes_position)==0 and len(lettres_mauvaises_position)==0:
        print("Aucune lettre correcte.")
    return lettres_bonnes_position, lettres_mauvaises_position
#endregion

# region Initialisation de jeu
longueur=input("Entrez la longueur du mot souhaitée: ")
# Choix au hasard d'un mot selon la longueur spécifiée dans une base de données
with open("liste_francais.txt", "r") as f:
    liste_mots=[mot.rstrip("\n").lower() for mot in f if len(mot.rstrip("\n"))==int(longueur)]
mot_aleatoire=liste_mots[random.randint(0,len(liste_mots))]
limit=0
# endregion

def valider():
    """
        Valide la saisie, met à jour la grille (lettres bien/mal placées) et gère victoires/erreurs.
        Vérifie la longueur et l'existence du mot, puis appelle comparaison() pour colorer les cases.
        Désactive l'entrée et le bouton en cas de victoire ou d'épuisement des tentatives.
    """
    global mot_aleatoire, limit
    mot = entree.get()
    entree.delete(0, tk.END)
    if limit < 6:
        if len(mot) != int(longueur):
            entree.insert(0, f"Le mot doit faire {longueur} lettres.") 
        elif len(mot) == int(longueur) and mot in liste_mots:
            lettres_bp, lettres_mp = comparaison(mot, mot_aleatoire)
            for j in range(cols):
                if j in lettres_bp:
                    grille[limit][j].config(text=mot[j], bg="green", fg="white")
                elif j in lettres_mp:
                    grille[limit][j].config(text=mot[j], bg="yellow", fg="black")
                else:
                    grille[limit][j].config(text=mot[j])
            limit += 1
            if mot == mot_aleatoire:
                bouton.config(state="disabled")
                texte = f"Félicitations! Vous avez deviné le mot"
                entree.insert(0, texte)
                entree.config(width=len(texte))
                entree.config(state="disabled")
            if limit == 6:
                entree.insert(0, f"Désolé, le mot était: {mot_aleatoire}")
        if len(mot) == int(longueur):
            if mot not in liste_mots:
                entree.insert(0, "Le mot n'existe pas dans la liste.")

# region Configuration de l'interface 
n = 6    
cols = int(longueur) 
cell_size = 4 

interface = tk.Tk()
interface.title("MyWordle")

cadre_grille = tk.Frame(interface, padx=10, pady=10) 
cadre_grille.pack()
#endregion 

# region Création de la grille de jeu
grille = []
for i in range(n):
    ligne = []
    for j in range(cols):
        lbl = tk.Label(cadre_grille, text=" ", width=cell_size, height=2, font=("Arial", 18), relief="solid", borderwidth=2, bg="white")
        lbl.grid(row=i, column=j, padx=0, pady=0, sticky="nsew")  
        ligne.append(lbl)
        cadre_grille.grid_columnconfigure(j, weight=1)
    grille.append(ligne)
    cadre_grille.grid_rowconfigure(i, weight=1)
#endregion

# Barre de saisie 
entree = tk.Entry(interface, font=("Arial", 14))
entree.pack(fill="x", padx=10, pady=5)

# Bouton de validation
bouton = tk.Button(interface, text="Valider", command=valider, font=("Arial", 12))
bouton.pack(padx=10, pady=5)
#endregion
interface.mainloop()
