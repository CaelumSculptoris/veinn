import os
import sys
import json
import argparse
from base64 import b64encode, b64decode
from src.utils.keygen import (generate_keypair)
from src.utils.keystore import (create_keystore, store_key_in_keystore)
from src.utils.menu import(
    menu_encrypt_with_pub, menu_decrypt_with_priv, menu_generate_keystore,
    menu_encrypt_with_public_veinn, menu_decrypt_with_public_veinn, menu_generate_keypair,
    menu_veinn_from_seed
    )
from src.models import (VeinnParams, bcolors)
from src.core import (encrypt_with_pub, decrypt_with_priv, veinn_from_seed)
def main():
    parser = argparse.ArgumentParser(description="VEINN - Vector Encrypted Invertible Neural Network")
    subparsers = parser.add_subparsers(dest="command")

    create_keystore_parser = subparsers.add_parser("create_keystore", help="Create encrypted keystore")
    create_keystore_parser.add_argument("--passphrase", required=True, help="Keystore passphrase")
    create_keystore_parser.add_argument("--keystore_file", default="keystore.json", help="Keystore filename")

    generate_parser = subparsers.add_parser("generate_keypair", help="Generate keypair")
    generate_parser.add_argument("--pubfile", default="public_key.json", help="Public key filename")
    generate_parser.add_argument("--privfile", default="private_key.json", help="Private key filename")
    generate_parser.add_argument("--keystore", default="keystore.json", help="Keystore filename")
    generate_parser.add_argument("--passphrase", help="Keystore passphrase")
    generate_parser.add_argument("--key_name", help="Key name in keystore")

    public_encrypt_parser = subparsers.add_parser("public_encrypt", help="Encrypt with public key")
    public_encrypt_parser.add_argument("--pubfile", default="public_key.json", help="Public key file")
    public_encrypt_parser.add_argument("--in_path", help="Input file path")
    public_encrypt_parser.add_argument("--file_type", choices=["json", "bin"], default="json", help="File type [JSON/BIN]")
    public_encrypt_parser.add_argument("--n", type=int, default=VeinnParams.n)
    public_encrypt_parser.add_argument("--rounds", type=int, default=VeinnParams.rounds)
    public_encrypt_parser.add_argument("--layers_per_round", type=int, default=VeinnParams.layers_per_round)
    public_encrypt_parser.add_argument("--shuffle_stride", type=int, default=VeinnParams.shuffle_stride)
    public_encrypt_parser.add_argument("--use_lwe", type=bool, default=True)
    public_encrypt_parser.add_argument("--q", type=int, default=VeinnParams.q)
    public_encrypt_parser.add_argument("--seed_len", type=int, default=32)
    public_encrypt_parser.add_argument("--nonce", help="Custom nonce (base64)")
    public_encrypt_parser.add_argument("--out_file", default="enc_pub")
    public_encrypt_parser.add_argument("--mode", default="cbc")

    public_decrypt_parser = subparsers.add_parser("public_decrypt", help="Decrypt with private key")
    public_decrypt_parser.add_argument("--keystore", default="keystore.json")
    public_decrypt_parser.add_argument("--privfile", default="private_key.json")
    public_decrypt_parser.add_argument("--encfile", default="enc_pub.json")
    public_decrypt_parser.add_argument("--passphrase")
    public_decrypt_parser.add_argument("--key_name")
    public_decrypt_parser.add_argument("--file_type", default="json")
    public_decrypt_parser.add_argument("--validity_window", type=int, default=3600)

    public_veinn_parser = subparsers.add_parser("public_veinn", help="Derive public VEINN from seed")
    public_veinn_parser.add_argument("--seed", required=True)
    public_veinn_parser.add_argument("--n", type=int, default=VeinnParams.n)
    public_veinn_parser.add_argument("--rounds", type=int, default=VeinnParams.rounds)
    public_veinn_parser.add_argument("--layers_per_round", type=int, default=VeinnParams.layers_per_round)
    public_veinn_parser.add_argument("--shuffle_stride", type=int, default=VeinnParams.shuffle_stride)
    public_veinn_parser.add_argument("--use_lwe", type=bool, default=True)
    public_veinn_parser.add_argument("--q", type=int, default=VeinnParams.q)

    args = parser.parse_known_args()[0]

    try:
        match args.command:
            case "create_keystore":
                create_keystore(args.passphrase, args.keystore_file)
                print(f"Keystore created: {args.keystore_file}")
            case "generate_keypair":
                keypair = generate_keypair()
                with open(args.pubfile, "w") as f:
                    json.dump({"ek": keypair["ek"]}, f)
                if args.keystore and args.passphrase and args.key_name:
                    store_key_in_keystore(args.passphrase, args.key_name, keypair, args.keystore)
                    print(f"Kyber keys generated: {args.pubfile} (public), private stored in keystore")
                else:
                    with open(args.privfile, "w") as f:
                        json.dump(keypair, f)
                    print(f"Kyber keys generated: {args.pubfile} (public), {args.privfile} (private)")
            case "public_encrypt":
                vp = VeinnParams(
                    n=args.n,
                    rounds=args.rounds,
                    layers_per_round=args.layers_per_round,
                    shuffle_stride=args.shuffle_stride,
                    use_lwe=args.use_lwe,
                    q=args.q
                )
                nonce = b64decode(args.nonce) if args.nonce else None
                encrypt_with_pub(
                    pubfile=args.pubfile,
                    file_type=args.file_type,
                    in_path=args.in_path,                    
                    vp=vp,
                    seed_len=args.seed_len,
                    nonce=nonce,
                    out_file=args.out_file,
                    mode=args.mode
                )
            case "public_decrypt":
                decrypt_with_priv(
                    keystore=args.keystore,
                    privfile=args.privfile,
                    encfile=args.encfile,
                    passphrase=args.passphrase,
                    key_name=args.key_name,
                    file_type=args.file_type,
                    validity_window=args.validity_window
                )
            case "public_veinn":
                vp = VeinnParams(
                    n=args.n,
                    rounds=args.rounds,
                    layers_per_round=args.layers_per_round,
                    shuffle_stride=args.shuffle_stride,
                    use_lwe=args.use_lwe,
                    q=args.q
                )
                veinn_from_seed(args.seed, vp)
            case _:
                _=os.system("cls") | os.system("clear")
                while True:
                    print(f"{bcolors.WARNING}{bcolors.BOLD}VEINN - Vector Encrypted Invertible Neural Network{bcolors.ENDC}")
                    print(f"{bcolors.GREY}{bcolors.BOLD}(]≡≡≡≡ø‡»{bcolors.OKCYAN}========================================-{bcolors.ENDC}")
                    print("")
                    print(f"{bcolors.BOLD}1){bcolors.ENDC} Create encrypted keystore")
                    print(f"{bcolors.BOLD}2){bcolors.ENDC} Generate keypair (public/private)")
                    print(f"{bcolors.BOLD}3){bcolors.ENDC} Encrypt with recipient public key")
                    print(f"{bcolors.BOLD}4){bcolors.ENDC} Decrypt with private key")
                    print(f"{bcolors.BOLD}5){bcolors.ENDC} Encrypt deterministically using public VEINN")
                    print(f"{bcolors.BOLD}6){bcolors.ENDC} Decrypt deterministically using public VEINN")
                    print(f"{bcolors.BOLD}7){bcolors.ENDC} Derive public VEINN from seed")                    
                    print(f"{bcolors.BOLD}0){bcolors.ENDC} Exit")
                    print("")
                    choice = input(f"{bcolors.BOLD}Choice: {bcolors.ENDC}").strip()
                    try:
                        match choice:
                            case "0":
                                break
                            case "1":
                                menu_generate_keystore()                                
                            case "2":
                                menu_generate_keypair()
                            case "3":
                                menu_encrypt_with_pub()
                            case "4":
                                menu_decrypt_with_priv()
                            case "5":
                                menu_encrypt_with_public_veinn()
                            case "6":
                                menu_decrypt_with_public_veinn()
                            case "7":
                                menu_veinn_from_seed()
                            case _:
                                print("Invalid choice")
                    except Exception as e:
                        print(f"{bcolors.FAIL}ERROR:{bcolors.ENDC}", e)
                    _=input(f"{bcolors.OKGREEN}Enter to continue...{bcolors.ENDC}")
                    _=os.system("cls") | os.system("clear")
    except Exception as e:
        print(f"{bcolors.FAIL}ERROR:{bcolors.ENDC}", e)
        sys.exit(1)

if __name__ == "__main__":
    main()