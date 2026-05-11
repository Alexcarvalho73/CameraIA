import os
import sys

# 1. Configura as variáveis de ambiente ANTES de importar o oracledb
instant_client_path = "/home/rdt/CameraIA/instantclient_21_1"
oracle_wallet_path = "/home/rdt/CameraIA/DriveOracle"

os.environ['LD_LIBRARY_PATH'] = f"{instant_client_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['TNS_ADMIN'] = oracle_wallet_path

import oracledb

def test_db():
    print("Iniciando teste de banco com IP DIRETO (Modo Thick)...")
    try:
        # Inicializa o cliente Oracle
        oracledb.init_oracle_client(lib_dir=instant_client_path)
        print("Oracle Thick Mode inicializado.")
        
        # DSN com IP direto para evitar problemas de DNS
        dsn_direto = '(description=(address=(protocol=tcps)(port=1522)(host=129.149.1.189))(connect_data=(service_name=g674a77dea23c6a_imaculado_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'
        
        print("Tentando conectar ao banco...")
        conn = oracledb.connect(
            user="mensagem",
            password="crbsAcs@2026",
            dsn=dsn_direto
        )
        print("Conexão estabelecida com sucesso!")
        
        cursor = conn.cursor()
        test_msg = "TESTE MANUAL - IP DIRETO"
        print(f"Tentando inserir registro: {test_msg}")
        
        sql_insert = "INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO) VALUES (:1, :2, :3, :4)"
        cursor.execute(sql_insert, ["5511999999999", test_msg, 0, 'G'])
        conn.commit()
        print("Inserção concluída.")
        
        cursor.close()
        conn.close()
        print("\nTeste finalizado com SUCESSO.")
        
    except Exception as e:
        print(f"\nERRO NO TESTE: {e}")

if __name__ == "__main__":
    test_db()
