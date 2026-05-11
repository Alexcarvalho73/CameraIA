import oracledb
import os

# Configurações
instant_client_path = "/home/rdt/CameraIA/instantclient_21_1"
oracle_wallet_path = "/home/rdt/CameraIA/DriveOracle"

def test_db_final():
    print("Iniciando teste de gravação e consulta direta (Modo Thick)...")
    try:
        oracledb.init_oracle_client(lib_dir=instant_client_path)
        # String simplificada sem retries para teste rápido
        full_dsn = '(description=(address=(protocol=tcps)(port=1522)(host=adb.sa-vinhedo-1.oraclecloud.com))(connect_data=(service_name=g674a77dea23c6a_imaculado_low.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'
        
        print(f"Tentando oracledb.connect()...")
        conn = oracledb.connect(
            user="mensagem",
            password="crbsAcs@2026",
            dsn=full_dsn,
            config_dir=oracle_wallet_path,
            wallet_location=oracle_wallet_path
        )
        print("Conexão estabelecida!")
        
        cursor = conn.cursor()
        
        # 1. Inserir teste
        cursor.execute("INSERT INTO DIZIMO.MENSAGENS (TELEFONE, TEXTO, STATUS, TIPO) VALUES (:1, :2, :3, :4)", 
                       ["5511000000000", "TESTE DE CONEXAO DIRETA", 1, 'G'])
        conn.commit()
        print("Registro inserido com sucesso.")
        
        # 2. SELECT solicitado pelo usuário
        print("\n--- RESULTADO DO SELECT * FROM DIZIMO.MENSAGENS (Top 5) ---")
        cursor.execute("SELECT * FROM (SELECT * FROM DIZIMO.MENSAGENS ORDER BY 1 DESC) WHERE ROWNUM <= 5")
        
        # Pega os nomes das colunas
        cols = [i[0] for i in cursor.description]
        print(" | ".join(cols))
        
        for row in cursor.fetchall():
            # Mostra apenas as primeiras colunas para não poluir (ignora o BLOB da imagem no print)
            print(" | ".join(str(val)[:50] for val in row[:5]))
            
        cursor.close()
        conn.close()
        print("\nSucesso total!")
        
    except Exception as e:
        print(f"\nERRO: {e}")

if __name__ == "__main__":
    test_db_final()
