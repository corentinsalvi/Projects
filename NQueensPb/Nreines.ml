(* Type pour représenter une position sur l'échiquier *)
type position = int * int 

(* Fonction pour vérifier si deux positions sont en conflit *)
let en_conflit (pos1 : position) (pos2 : position) : bool =
  let x1, y1 = pos1 in
  let x2, y2 = pos2 in
  x1 = x2 || y1 = y2 || abs (x1 - x2) = abs (y1 - y2)

(* Fonction récursive pour placer une reine sur l'échiquier *)
let rec placer_reine (n : int) (echiquier : position list) : position list option =
  match n with
  | 0 -> Some echiquier
  | _ ->
      let colonne = List.length echiquier in
      let candidats = List.init 10 (fun i -> (i, colonne)) in
      let rec loop = function
        | [] -> None
        | pos :: rest ->
            if List.exists (fun pos' -> en_conflit pos pos') echiquier then
              loop rest
            else
              match placer_reine (n - 1) (pos :: echiquier) with
              | None -> loop rest
              | Some solution -> Some solution
      in
      loop candidats

(* Fonction pour afficher la solution *)
let afficher_solution (solution : position list) : unit =
  let ligne y =
    String.concat "" (List.init 10 (fun x -> if List.mem (x, y) solution then "Q " else "- "))
  in
  let lignes = List.init 10 ligne in
  let echiquier = String.concat "\n" lignes in
  print_endline echiquier

(* Fonction principale pour résoudre le problème *)
let resoudre () =
  match placer_reine 10 [] with
  | None -> print_endline "Aucune solution trouvée."
  | Some solution -> afficher_solution solution

(* Exécution *)
let () = resoudre () ;;